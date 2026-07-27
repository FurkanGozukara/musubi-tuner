import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from musubi_tuner.torch_compile_toolchain.compiler_probe import CompilerProbeStatus
from musubi_tuner.torch_compile_toolchain.cuda_toolkit import _looks_like_cuda_root
from musubi_tuner.torch_compile_toolchain.discovery import (
    CudaDiscoveryStatus,
    ExecutableStatus,
    discover_cuda_environment,
    discover_ninja,
    discover_posix_compiler,
)
from musubi_tuner.torch_compile_toolchain.environment import (
    CompileToolchainStatus,
    ensure_compile_environment,
    prepare_compile_subprocess_env,
)
from musubi_tuner.torch_compile_toolchain.host_compiler_probe import (
    HostCompilerProbeStatus,
    probe_host_cpp_compiler,
)
from musubi_tuner.torch_compile_toolchain.msvc import (
    MsvcLoadStatus,
    _environment_from_vc_script,
    _scripts_for_vs_root,
    _sort_msvc_candidates,
    load_msvc_environment,
)
from musubi_tuner.torch_compile_toolchain import msvc
from musubi_tuner.torch_compile_toolchain.runtime import (
    compile_callable,
    compile_module_callable,
)
from musubi_tuner.torch_compile_toolchain import example
from musubi_tuner.training.compile_setup import (
    TorchCompileToolchainError,
    compile_requested,
    disable_unavailable_dynamo_backend,
    ensure_training_compile_environment,
    native_compile_toolchain_requested,
)
from musubi_tuner.training import compile_setup as training_compile_setup
from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader, Offloader
from musubi_tuner.utils import model_utils


def _touch_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_cuda_root(path: Path) -> Path:
    cuda_bin = path / "bin"
    cuda_bin.mkdir(parents=True)
    _touch_executable(cuda_bin / "nvcc")
    _touch_executable(cuda_bin / "nvcc.exe")
    return cuda_bin


class CudaAndNinjaDiscoveryTests(unittest.TestCase):
    def test_generic_lib64_directory_is_not_cuda(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "lib64").mkdir()
            self.assertFalse(_looks_like_cuda_root(root, "linux"))

    def test_cuda_selection_matches_torch_instead_of_stale_cuda_home(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_12 = Path(temp_dir) / "v12.8"
            cuda_13 = Path(temp_dir) / "v13.0"
            _make_cuda_root(cuda_12)
            cuda_13_bin = _make_cuda_root(cuda_13)
            env = {
                "CUDA_HOME": str(cuda_12),
                "CUDA_PATH": str(cuda_13),
                "PATH": "",
            }
            with (
                patch(
                    "musubi_tuner.torch_compile_toolchain.discovery.sys.platform",
                    "linux",
                ),
                patch(
                    "musubi_tuner.torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                    return_value="13.0",
                ),
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cuda_13.resolve()), env["CUDA_HOME"])
        self.assertEqual(str(cuda_13.resolve()), env["CUDA_PATH"])
        self.assertEqual(str(cuda_13_bin), env["PATH"].split(os.pathsep)[0])

    def test_cuda_selection_uses_nearest_same_major_minor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_13_2 = Path(temp_dir) / "v13.2"
            cuda_13_1 = Path(temp_dir) / "v13.1"
            _make_cuda_root(cuda_13_2)
            _make_cuda_root(cuda_13_1)
            env = {
                "CUDA_HOME": str(cuda_13_2),
                "CUDA_PATH": str(cuda_13_1),
                "PATH": "",
            }
            with (
                patch(
                    "musubi_tuner.torch_compile_toolchain.discovery.sys.platform",
                    "linux",
                ),
                patch(
                    "musubi_tuner.torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                    return_value="13.0",
                ),
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cuda_13_1.resolve()), env["CUDA_HOME"])

    def test_versioned_cuda_installer_variable_is_discovered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_root = Path(temp_dir) / "v13.1"
            _make_cuda_root(cuda_root)
            env = {"CUDA_PATH_V13_1": str(cuda_root), "PATH": ""}
            with (
                patch(
                    "musubi_tuner.torch_compile_toolchain.discovery.sys.platform",
                    "win32",
                ),
                patch(
                    "musubi_tuner.torch_compile_toolchain.cuda_toolkit.torch_cuda_version",
                    return_value="13.0",
                ),
            ):
                status = discover_cuda_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(cuda_root.resolve()), env["CUDA_PATH"])

    def test_ninja_is_found_in_virtual_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scripts = Path(temp_dir) / "Scripts"
            ninja = scripts / "ninja.exe"
            _touch_executable(ninja)
            env = {"VIRTUAL_ENV": temp_dir, "PATH": ""}
            with patch(
                "musubi_tuner.torch_compile_toolchain.discovery.sys.platform",
                "win32",
            ):
                status = discover_ninja(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(ninja.resolve()), env["NINJA"])

    def test_ninja_is_found_in_custom_visual_studio_install(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            install = Path(temp_dir) / "CustomVS" / "BuildTools"
            ninja = install / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe"
            _touch_executable(ninja)
            env = {"PATH": ""}
            with (
                patch(
                    "musubi_tuner.torch_compile_toolchain.discovery.sys.platform",
                    "win32",
                ),
                patch(
                    "musubi_tuner.torch_compile_toolchain.discovery.visual_studio_install_roots",
                    return_value=[install],
                ),
            ):
                status = discover_ninja(env)

        self.assertTrue(status.ok)
        self.assertEqual(str(ninja.resolve()), env["NINJA"])


class HostCompilerProbeTests(unittest.TestCase):
    def test_failed_standard_header_compile_is_not_ready(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="probe.cpp(1): fatal error C1034: vector: no include path set",
            stderr="",
        )
        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.host_compiler_probe._resolve_compiler",
                return_value=r"C:\VS\cl.exe",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.host_compiler_probe.subprocess.run",
                return_value=completed,
            ),
        ):
            status = probe_host_cpp_compiler(
                {"PATH": r"C:\VS"},
                platform_name="win32",
                require_openmp=True,
            )

        self.assertFalse(status.ok)
        self.assertIn("no include path", status.detail)

    def test_successful_compile_and_execution_is_ready(self):
        def run(command, **_kwargs):
            if any(str(item).startswith("/Fe:") for item in command):
                output = next(str(item)[4:] for item in command if str(item).startswith("/Fe:"))
                Path(output).touch()
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.host_compiler_probe._resolve_compiler",
                return_value=r"C:\VS\cl.exe",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.host_compiler_probe.subprocess.run",
                side_effect=run,
            ),
        ):
            status = probe_host_cpp_compiler(
                {"PATH": r"C:\VS"},
                platform_name="win32",
                require_openmp=True,
            )

        self.assertTrue(status.ok)
        self.assertIn("OpenMP", status.detail)


class WindowsDiscoveryTests(unittest.TestCase):
    def test_newest_installed_toolset_is_first_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _touch_executable(root / "Common7" / "Tools" / "VsDevCmd.bat")
            for version in ("14.29.30133", "14.50.10000"):
                _touch_executable(root / "VC" / "Tools" / "MSVC" / version / "bin" / "Hostx64" / "x64" / "cl.exe")
            scripts = _scripts_for_vs_root(root)

        self.assertIn("-vcvars_ver=14.50.10000", scripts[0][1])
        self.assertTrue(any("-vcvars_ver=14.50" in args for _, args in scripts))

    def test_candidates_sort_by_toolset_not_installation_name(self):
        older = (Path(r"C:\VS2022\VsDevCmd.bat"), "-vcvars_ver=14.44.35207")
        newer = (Path(r"C:\VS18\VsDevCmd.bat"), "-vcvars_ver=14.50.10000")
        self.assertEqual(newer, _sort_msvc_candidates([older, newer])[0])

    def test_legacy_vcvarsall_layout_remains_a_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_script = root / "VC" / "vcvarsall.bat"
            _touch_executable(legacy_script)
            scripts = _scripts_for_vs_root(root)

        self.assertIn((legacy_script, "amd64"), scripts)

    def test_explicit_developer_script_is_tried_before_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured = root / "configured.bat"
            discovered = root / "discovered.bat"
            configured.touch()
            discovered.touch()
            with (
                patch.object(msvc, "visual_studio_install_roots", return_value=[root]),
                patch.object(
                    msvc,
                    "_scripts_for_vs_root",
                    return_value=[(discovered, "-vcvars_ver=14.44")],
                ),
            ):
                candidates = msvc._candidate_msvc_scripts({"MUSUBI_VS_DEV_CMD": str(configured)})

        self.assertEqual(configured, candidates[0][0])
        self.assertEqual(discovered, candidates[1][0])

    def test_vc_script_uses_cmd_compatible_quoting(self):
        script = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VsDevCmd.bat")
        completed = subprocess.CompletedProcess(
            args="",
            returncode=0,
            stdout="Path=C:\\VS\\bin\nVCToolsVersion=14.44.35207\n",
            stderr="",
        )
        with patch(
            "musubi_tuner.torch_compile_toolchain.msvc.subprocess.run",
            return_value=completed,
        ) as run:
            loaded = _environment_from_vc_script(
                script,
                "-arch=amd64 -host_arch=amd64",
                base_env={"COMSPEC": r"C:\Windows\System32\cmd.exe", "PATH": ""},
            )

        command_line = run.call_args.args[0]
        self.assertIsInstance(command_line, str)
        self.assertIn(f'call "{script}"', command_line)
        self.assertNotIn(r"\"C:\Program Files", command_line)
        self.assertEqual(r"C:\VS\bin", loaded["Path"])

    def test_mixed_case_path_from_cmd_replaces_existing_path(self):
        script = Path(r"C:\VS\Common7\Tools\VsDevCmd.bat")
        env = {"PATH": "original"}

        def has_loaded_cl(candidate):
            return candidate.get("PATH") == "developer-path" and "Path" not in candidate

        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc._candidate_msvc_scripts",
                return_value=[(script, "")],
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc._environment_from_vc_script",
                return_value={
                    "Path": "developer-path",
                    "VCToolsVersion": "14.44.35207",
                },
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.has_cl_exe",
                side_effect=has_loaded_cl,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.probe_host_cpp_compiler",
                return_value=HostCompilerProbeStatus(True, "host probe passed"),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA probe passed"),
            ),
        ):
            status = load_msvc_environment(env)

        self.assertTrue(status.ok)
        self.assertEqual("developer-path", env["PATH"])
        self.assertNotIn("Path", env)

    def test_partial_path_compiler_activates_developer_environment(self):
        env = {"PATH": r"C:\VS\bin"}

        def load(candidate, **_kwargs):
            candidate["INCLUDE"] = r"C:\VS\include"
            candidate["LIB"] = r"C:\VS\lib"
            return MsvcLoadStatus(True, "MSVC environment loaded", True)

        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.sys.platform",
                "win32",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_cuda_environment",
                return_value=CudaDiscoveryStatus(True, "CUDA ready", r"C:\CUDA", False),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.has_cl_exe",
                return_value=True,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.probe_host_cpp_compiler",
                return_value=HostCompilerProbeStatus(False, "standard headers missing"),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA probe passed"),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.load_msvc_environment",
                side_effect=load,
            ) as load_msvc,
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_ninja",
                return_value=ExecutableStatus(True, "ninja ready", r"C:\ninja.exe"),
            ),
        ):
            status = ensure_compile_environment(env, require_openmp=True)

        self.assertTrue(status.ok)
        load_msvc.assert_called_once_with(env, require_openmp=True)
        self.assertIn("INCLUDE", env)
        self.assertIn("LIB", env)

    def test_msvc_loader_skips_incomplete_toolset(self):
        scripts = [(Path("first.bat"), "14.50"), (Path("second.bat"), "14.44")]

        def script_env(_script, args, **_kwargs):
            return {"PATH": args, "VCToolsVersion": args}

        env = {"PATH": ""}
        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc._candidate_msvc_scripts",
                return_value=scripts,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc._environment_from_vc_script",
                side_effect=script_env,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.has_cl_exe",
                return_value=True,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.shutil.which",
                return_value=r"C:\VS\cl.exe",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.probe_host_cpp_compiler",
                side_effect=[
                    HostCompilerProbeStatus(False, "headers missing"),
                    HostCompilerProbeStatus(True, "host probe passed"),
                ],
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.msvc.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA probe passed"),
            ),
        ):
            status = load_msvc_environment(env, require_openmp=True)

        self.assertTrue(status.ok)
        self.assertEqual("14.44", env["VCToolsVersion"])


class PosixDiscoveryTests(unittest.TestCase):
    def test_linux_environment_requires_a_probed_openmp_compiler(self):
        env = {"PATH": "/usr/bin"}
        compiler_status = ExecutableStatus(
            True,
            "g++ ready with OpenMP",
            "/usr/bin/g++",
        )
        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.sys.platform",
                "linux",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_cuda_environment",
                return_value=CudaDiscoveryStatus(False, "CUDA Toolkit not found"),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_posix_compiler",
                return_value=compiler_status,
            ) as discover_compiler,
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_ninja",
                return_value=ExecutableStatus(True, "ninja ready", "/usr/bin/ninja"),
            ),
        ):
            status = ensure_compile_environment(env, require_openmp=True)

        self.assertTrue(status.ok)
        self.assertEqual("linux", status.platform)
        discover_compiler.assert_called_once_with(env, require_openmp=True)

    def test_linux_discovery_skips_host_compiler_that_cannot_link(self):
        env = {"PATH": "/tools"}

        def which(name, **_kwargs):
            return {"g++-14": "/tools/g++-14", "g++-13": "/tools/g++-13"}.get(name, "")

        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.discovery.sys.platform",
                "linux",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.discovery._posix_compiler_candidates",
                return_value=(("g++-14", ""), ("g++-13", "")),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.discovery._prepend_existing_paths",
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.discovery.shutil.which",
                side_effect=which,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.discovery.probe_host_cpp_compiler",
                side_effect=[
                    HostCompilerProbeStatus(False, "OpenMP unavailable"),
                    HostCompilerProbeStatus(True, "host probe passed with OpenMP"),
                ],
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.discovery.probe_cuda_compiler",
                return_value=CompilerProbeStatus(True, "CUDA probe passed"),
            ),
        ):
            status = discover_posix_compiler(env, require_openmp=True)

        self.assertTrue(status.ok)
        self.assertEqual("/tools/g++-13", env["CXX"])
        self.assertEqual("/tools/g++-13", env["CUDAHOSTCXX"])


class ProbeVerificationCacheTests(unittest.TestCase):
    def _paths(self, root: Path) -> tuple[Path, Path, Path, Path]:
        compiler = root / "msvc" / "cl.exe"
        ninja = root / "tools" / "ninja.exe"
        cuda_root = root / "cuda"
        nvcc = cuda_root / "bin" / "nvcc.exe"
        for path in (compiler, ninja, nvcc):
            _touch_executable(path)
        return compiler, ninja, cuda_root, nvcc

    def _ensure_windows(
        self,
        env,
        *,
        compiler,
        ninja,
        cuda_root,
        nvcc,
        cache_dir,
        forbid_probes=False,
        force_probe=False,
    ):
        def discover_cuda(candidate):
            candidate["CUDA_HOME"] = str(cuda_root)
            candidate["CUDA_PATH"] = str(cuda_root)
            candidate["CUDACXX"] = str(nvcc)
            return CudaDiscoveryStatus(True, "CUDA ready", str(cuda_root), True)

        def discover_ninja_mock(candidate):
            candidate["NINJA"] = str(ninja)
            return ExecutableStatus(True, "ninja ready", str(ninja))

        probe_effect = AssertionError("native probe should have been skipped") if forbid_probes else None
        with (
            patch("musubi_tuner.torch_compile_toolchain.environment.sys.platform", "win32"),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_cuda_environment",
                side_effect=discover_cuda if not forbid_probes else probe_effect,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.discover_ninja",
                side_effect=discover_ninja_mock if not forbid_probes else probe_effect,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.has_cl_exe",
                side_effect=probe_effect if forbid_probes else None,
                return_value=True,
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.probe_host_cpp_compiler",
                side_effect=probe_effect,
                return_value=HostCompilerProbeStatus(True, "host probe passed with OpenMP"),
            ) as host_probe,
            patch(
                "musubi_tuner.torch_compile_toolchain.environment.probe_cuda_compiler",
                side_effect=probe_effect,
                return_value=CompilerProbeStatus(True, "CUDA probe passed"),
            ) as cuda_probe,
            patch(
                "musubi_tuner.torch_compile_toolchain.environment._compiler_on_path",
                return_value=str(compiler),
            ),
            patch(
                "musubi_tuner.torch_compile_toolchain.readiness_cache.torch_cuda_version",
                return_value="13.1",
            ),
        ):
            status = ensure_compile_environment(
                env,
                cache_dir=cache_dir,
                require_openmp=True,
                force_probe=force_probe,
            )
        return status, host_probe, cuda_probe

    def test_successful_probe_is_reused_from_disk_without_running_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler, ninja, cuda_root, nvcc = self._paths(root)
            cache_dir = root / "cache"
            base = {"PATH": "base", "AWS_SECRET_ACCESS_KEY": "do-not-persist"}

            first_env = dict(base)
            first, host_probe, cuda_probe = self._ensure_windows(
                first_env,
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
            )
            cache_text = (cache_dir / "toolchain_probe_v1.json").read_text(encoding="utf-8")

            second_env = dict(base)
            second, skipped_host, skipped_cuda = self._ensure_windows(
                second_env,
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
                forbid_probes=True,
            )

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(1, host_probe.call_count)
        self.assertEqual(1, cuda_probe.call_count)
        skipped_host.assert_not_called()
        skipped_cuda.assert_not_called()
        self.assertIn("toolchain probe cache hit", second.detail)
        self.assertEqual(str(cuda_root), second_env["CUDA_HOME"])
        self.assertEqual(str(ninja), second_env["NINJA"])
        self.assertNotIn("do-not-persist", cache_text)

    def test_inherited_verified_environment_skips_child_process_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler, ninja, cuda_root, nvcc = self._paths(root)
            cache_dir = root / "cache"
            parent_env = {"PATH": "base"}
            first, _, _ = self._ensure_windows(
                parent_env,
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
            )
            parent_env["MUSUBI_TORCH_COMPILE_DETAIL"] = first.detail
            (cache_dir / "toolchain_probe_v1.json").unlink()

            child_env = dict(parent_env)
            child, skipped_host, skipped_cuda = self._ensure_windows(
                child_env,
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
                forbid_probes=True,
            )

        self.assertTrue(child.ok)
        skipped_host.assert_not_called()
        skipped_cuda.assert_not_called()
        self.assertIn("inherited toolchain verification reused", child.detail)

    def test_changed_compiler_invalidates_persisted_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler, ninja, cuda_root, nvcc = self._paths(root)
            cache_dir = root / "cache"
            base = {"PATH": "base"}
            self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
            )
            compiler.write_bytes(b"changed compiler")

            refreshed, host_probe, cuda_probe = self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
            )

        self.assertTrue(refreshed.ok)
        self.assertNotIn("toolchain probe cache hit", refreshed.detail)
        self.assertEqual(1, host_probe.call_count)
        self.assertEqual(1, cuda_probe.call_count)

    def test_force_probe_bypasses_a_valid_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler, ninja, cuda_root, nvcc = self._paths(root)
            cache_dir = root / "cache"
            base = {"PATH": "base"}
            self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
            )

            refreshed, host_probe, cuda_probe = self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=cache_dir,
                force_probe=True,
            )

        self.assertTrue(refreshed.ok)
        self.assertNotIn("toolchain probe cache hit", refreshed.detail)
        self.assertEqual(1, host_probe.call_count)
        self.assertEqual(1, cuda_probe.call_count)

    def test_validation_cache_stays_install_local_when_artifact_cache_is_redirected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            compiler, ninja, cuda_root, nvcc = self._paths(root)
            install_cache = root / "fresh-install" / ".cache" / "torch_compile"
            artifact_cache = root / "shared-artifacts"
            (artifact_cache / "inductor").mkdir(parents=True)
            (artifact_cache / "triton").mkdir(parents=True)
            base = {
                "PATH": "base",
                "TORCHINDUCTOR_CACHE_DIR": str(artifact_cache / "inductor"),
                "TRITON_CACHE_DIR": str(artifact_cache / "triton"),
            }

            first, _, _ = self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=install_cache,
            )
            second, skipped_host, skipped_cuda = self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=install_cache,
                forbid_probes=True,
            )
            fresh_install_cache = root / "second-fresh-install" / ".cache" / "torch_compile"
            fresh_install, fresh_host, fresh_cuda = self._ensure_windows(
                dict(base),
                compiler=compiler,
                ninja=ninja,
                cuda_root=cuda_root,
                nvcc=nvcc,
                cache_dir=fresh_install_cache,
            )
            install_cache_exists = (install_cache / "toolchain_probe_v1.json").is_file()
            fresh_install_cache_exists = (fresh_install_cache / "toolchain_probe_v1.json").is_file()
            shared_cache_exists = (artifact_cache / "toolchain_probe_v1.json").exists()

        self.assertEqual(str(artifact_cache), first.cache_root)
        self.assertTrue(install_cache_exists)
        self.assertTrue(fresh_install.ok)
        self.assertNotIn("toolchain probe cache hit", fresh_install.detail)
        self.assertTrue(fresh_install_cache_exists)
        self.assertEqual(1, fresh_host.call_count)
        self.assertEqual(1, fresh_cuda.call_count)
        self.assertFalse(shared_cache_exists)
        self.assertIn("toolchain probe cache hit", second.detail)
        skipped_host.assert_not_called()
        skipped_cuda.assert_not_called()


class SubprocessAndRequestTests(unittest.TestCase):
    def test_example_uses_repository_root(self):
        self.assertEqual(Path(__file__).resolve().parents[1], example.PROJECT_ROOT)

    def test_all_accelerator_entry_points_prepare_inductor_toolchain(self):
        source_root = Path(__file__).resolve().parents[1] / "src" / "musubi_tuner"
        for relative_path in (
            Path("training") / "accelerator_setup.py",
            Path("hv_train.py"),
        ):
            with self.subTest(path=relative_path):
                source = (source_root / relative_path).read_text(encoding="utf-8")
                self.assertIn("native_compile_toolchain_requested(args)", source)
                self.assertIn("ensure_training_compile_environment()", source)

    def test_compile_environment_is_not_touched_when_not_requested(self):
        source = {"PATH": "original"}
        with patch("musubi_tuner.torch_compile_toolchain.environment.ensure_compile_environment") as ensure:
            child = prepare_compile_subprocess_env(source, compile_requested=False)

        ensure.assert_not_called()
        self.assertEqual(source, child)
        self.assertIsNot(source, child)

    def test_compile_request_detects_direct_and_dynamo_modes(self):
        class Args:
            compile = False
            dynamo_backend = "NO"

        self.assertFalse(compile_requested(Args()))
        Args.compile = True
        self.assertTrue(compile_requested(Args()))
        Args.compile = False
        Args.dynamo_backend = "inductor"
        self.assertTrue(compile_requested(Args()))

    def test_accelerate_environment_requests_setup(self):
        class Args:
            compile = False
            dynamo_backend = "NO"

        with patch.dict(os.environ, {"ACCELERATE_DYNAMO_BACKEND": "inductor"}, clear=False):
            self.assertTrue(compile_requested(Args()))

    def test_only_inductor_requests_native_toolchain(self):
        args = SimpleNamespace(
            compile=True,
            compile_backend="eager",
            dynamo_backend="NO",
        )
        self.assertFalse(native_compile_toolchain_requested(args))
        args.compile_backend = "inductor"
        self.assertTrue(native_compile_toolchain_requested(args))
        args.compile = False
        args.dynamo_backend = "aot_eager"
        self.assertFalse(native_compile_toolchain_requested(args))
        args.dynamo_backend = "inductor"
        self.assertTrue(native_compile_toolchain_requested(args))


class TrainingCompileFallbackTests(unittest.TestCase):
    def setUp(self):
        self.original_status = training_compile_setup._ready_status
        training_compile_setup._ready_status = None

    def tearDown(self):
        training_compile_setup._ready_status = self.original_status

    def test_unavailable_toolchain_defaults_to_nonfatal_eager_fallback(self):
        unavailable = CompileToolchainStatus(False, "synthetic compiler failure")
        with (
            patch(
                "musubi_tuner.training.compile_setup.ensure_compile_environment",
                return_value=unavailable,
            ) as ensure,
            patch.dict(os.environ, {}, clear=False),
        ):
            status = ensure_training_compile_environment()
            cached_status = ensure_training_compile_environment()
            ready = os.environ["MUSUBI_TORCH_COMPILE_READY"]
            active = os.environ["MUSUBI_TORCH_COMPILE_ACTIVE"]

        self.assertIs(unavailable, status)
        self.assertIs(unavailable, cached_status)
        self.assertEqual(1, ensure.call_count)
        self.assertEqual("0", ready)
        self.assertEqual("0", active)

    def test_strict_toolchain_mode_can_still_raise(self):
        unavailable = CompileToolchainStatus(False, "synthetic compiler failure")
        with patch(
            "musubi_tuner.training.compile_setup.ensure_compile_environment",
            return_value=unavailable,
        ):
            ensure_training_compile_environment()
            with self.assertRaises(TorchCompileToolchainError):
                ensure_training_compile_environment(required=True)

    def test_unavailable_toolchain_disables_accelerate_inductor_backend(self):
        args = SimpleNamespace(dynamo_backend="inductor")
        status = CompileToolchainStatus(False, "unavailable")
        with patch.dict(os.environ, {"ACCELERATE_DYNAMO_BACKEND": "inductor"}, clear=False):
            disable_unavailable_dynamo_backend(args, status)
            self.assertEqual("NO", os.environ["ACCELERATE_DYNAMO_BACKEND"])
        self.assertEqual("NO", args.dynamo_backend)


class PortableCompileRuntimeTests(unittest.TestCase):
    def test_disabled_compile_returns_original_callable(self):
        def target(value):
            return value + 1

        result = compile_callable(target, enabled=False)

        self.assertFalse(result.requested)
        self.assertFalse(result.compiled)
        self.assertIs(target, result.callable)

    def test_eager_backend_skips_native_toolchain_discovery(self):
        def target(value):
            return value + 1

        with (
            patch("musubi_tuner.torch_compile_toolchain.runtime.ensure_compile_environment") as ensure,
            patch("torch.compile", side_effect=lambda callable_target, **_kwargs: callable_target),
        ):
            result = compile_callable(target, backend="eager")
            output = result.callable(2)

        ensure.assert_not_called()
        self.assertEqual(3, output)
        self.assertTrue(result.verified)

    def test_first_compiled_call_marks_result_verified(self):
        def target(value):
            return value + 1

        def compile_target(callable_target, **_kwargs):
            return lambda value: callable_target(value)

        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.runtime.ensure_compile_environment",
                return_value=CompileToolchainStatus(True, "ready"),
            ),
            patch("torch.compile", side_effect=compile_target),
        ):
            result = compile_callable(target)
            output = result.callable(2)

        self.assertEqual(3, output)
        self.assertTrue(result.compiled)
        self.assertTrue(result.verified)

    def test_failed_cold_compile_call_falls_back_to_eager(self):
        def target(value):
            return value + 1

        def compile_target(_target, **_kwargs):
            def failed(_value):
                raise RuntimeError("compiler failure")

            return failed

        with (
            patch(
                "musubi_tuner.torch_compile_toolchain.runtime.ensure_compile_environment",
                return_value=CompileToolchainStatus(True, "ready"),
            ),
            patch("torch.compile", side_effect=compile_target),
        ):
            result = compile_callable(target)
            first = result.callable(2)
            second = result.callable(4)

        self.assertEqual(3, first)
        self.assertEqual(5, second)
        self.assertFalse(result.compiled)
        self.assertFalse(result.verified)
        self.assertIn("eager fallback", result.detail)

    def test_module_callable_is_replaced_in_place(self):
        module = SimpleNamespace(forward=lambda value: value * 2)
        replacement = SimpleNamespace(callable=lambda value: value * 3)
        with patch(
            "musubi_tuner.torch_compile_toolchain.runtime.compile_callable",
            return_value=replacement,
        ):
            result = compile_module_callable(module)

        self.assertEqual(9, module.forward(3))
        self.assertIs(result.callable, module.forward)


class CompiledBlockFallbackTests(unittest.TestCase):
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

    class FailingCompiledBlock(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original

        def forward(self, _value):
            raise RuntimeError("synthetic cold compile failure")

    class PassingCompiledBlock(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original

        def forward(self, value):
            return self._orig_mod(value)

    def test_first_call_failure_falls_back_without_changing_state_dict_shape(self):
        transformer = self.Transformer()
        original = transformer.blocks[0]
        args = SimpleNamespace(
            compile_dynamic="auto",
            compile_backend="inductor",
            compile_mode="default",
            compile_fullgraph=False,
            compile_cache_size_limit=None,
        )

        with (
            patch("musubi_tuner.utils.model_utils.ensure_training_compile_environment"),
            patch(
                "musubi_tuner.utils.model_utils.torch.compile",
                side_effect=lambda block, **_kwargs: self.FailingCompiledBlock(block),
            ),
            patch.dict(os.environ, {"MUSUBI_TORCH_COMPILE_FALLBACK": "1"}, clear=False),
        ):
            compiled = model_utils.compile_transformer(
                args,
                transformer,
                [transformer.blocks],
                disable_linear=False,
            )
            value = torch.ones(1, 2)
            output = compiled.blocks[0](value)

        torch.testing.assert_close(output, original(value))
        self.assertFalse(compiled._musubi_compile_state["enabled"])
        self.assertIn("blocks.0._orig_mod.weight", compiled.state_dict())

    def test_unavailable_toolchain_leaves_transformer_eager(self):
        transformer = self.Transformer()
        original = transformer.blocks[0]
        args = SimpleNamespace(
            compile_dynamic="auto",
            compile_backend="inductor",
            compile_mode="default",
            compile_fullgraph=False,
            compile_cache_size_limit=None,
        )
        unavailable = CompileToolchainStatus(False, "synthetic toolchain failure")

        with (
            patch(
                "musubi_tuner.utils.model_utils.ensure_training_compile_environment",
                return_value=unavailable,
            ),
            patch("musubi_tuner.utils.model_utils.torch.compile") as compile_mock,
            patch.dict(os.environ, {"MUSUBI_TORCH_COMPILE_FALLBACK": "1"}, clear=False),
        ):
            result = model_utils.compile_transformer(
                args,
                transformer,
                [transformer.blocks],
                disable_linear=False,
            )

        self.assertIs(transformer, result)
        self.assertIs(original, result.blocks[0])
        compile_mock.assert_not_called()
        self.assertFalse(result._musubi_compile_state["enabled"])
        self.assertIn("toolchain unavailable", result._musubi_compile_state["fallback_reason"])

    def test_torch_compile_setup_exception_leaves_transformer_eager(self):
        transformer = self.Transformer()
        original = transformer.blocks[0]
        args = SimpleNamespace(
            compile_dynamic="auto",
            compile_backend="inductor",
            compile_mode="default",
            compile_fullgraph=False,
            compile_cache_size_limit=None,
        )

        with (
            patch(
                "musubi_tuner.utils.model_utils.ensure_training_compile_environment",
                return_value=CompileToolchainStatus(True, "ready"),
            ),
            patch(
                "musubi_tuner.utils.model_utils.torch.compile",
                side_effect=RuntimeError("synthetic setup failure"),
            ),
            patch.dict(os.environ, {"MUSUBI_TORCH_COMPILE_FALLBACK": "1"}, clear=False),
        ):
            result = model_utils.compile_transformer(
                args,
                transformer,
                [transformer.blocks],
                disable_linear=False,
            )

        self.assertIs(original, result.blocks[0])
        self.assertFalse(result._musubi_compile_state["enabled"])
        self.assertIn("setup failed", result._musubi_compile_state["fallback_reason"])

    def test_fallback_disabled_still_records_verified_compiled_blocks(self):
        transformer = self.Transformer()
        args = SimpleNamespace(
            compile_dynamic="auto",
            compile_backend="inductor",
            compile_mode="default",
            compile_fullgraph=False,
            compile_cache_size_limit=None,
        )

        with (
            patch("musubi_tuner.utils.model_utils.ensure_training_compile_environment"),
            patch(
                "musubi_tuner.utils.model_utils.torch.compile",
                side_effect=lambda block, **_kwargs: self.PassingCompiledBlock(block),
            ),
            patch.dict(os.environ, {"MUSUBI_TORCH_COMPILE_FALLBACK": "0"}, clear=False),
        ):
            compiled = model_utils.compile_transformer(
                args,
                transformer,
                [transformer.blocks],
                disable_linear=False,
            )
            compiled.blocks[0](torch.ones(1, 2))

        self.assertEqual(1, len(compiled._musubi_compile_state["verified"]))
        self.assertTrue(compiled._musubi_compile_state["enabled"])


class ResidentBlockCompileTests(unittest.TestCase):
    class Transformer(torch.nn.Module):
        def __init__(self, count=3):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.ReLU()) for _ in range(count)]
            )

    class CompiledBlock(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original

        def forward(self, value):
            return self._orig_mod(value)

    @staticmethod
    def args():
        return SimpleNamespace(
            compile_dynamic="auto",
            compile_backend="inductor",
            compile_mode="default",
            compile_fullgraph=False,
            compile_cache_size_limit=None,
            compile_resident_blocks_only=True,
        )

    def test_classic_offloader_is_conservative_at_cyclic_sampling_boundary(self):
        offloader = object.__new__(Offloader)
        offloader.num_blocks = 5
        offloader.blocks_to_swap = 1
        self.assertEqual([1, 2, 3], offloader.compile_safe_block_indices())
        offloader.blocks_to_swap = 2
        self.assertEqual([], offloader.compile_safe_block_indices())

    def test_stream_offloader_returns_non_streaming_complement(self):
        offloader = object.__new__(LoRAStreamOffloader)
        offloader.is_stream = [True, False, True, False]
        self.assertEqual([1, 3], offloader.compile_safe_block_indices())

    def test_only_resident_blocks_are_wrapped_and_linears_stay_compile_eligible(self):
        transformer = self.Transformer()
        originals = list(transformer.blocks)
        offloader = SimpleNamespace(compile_safe_block_indices=lambda: [1])

        with (
            patch("musubi_tuner.utils.model_utils.ensure_training_compile_environment"),
            patch(
                "musubi_tuner.utils.model_utils.torch.compile",
                side_effect=lambda block, **_kwargs: self.CompiledBlock(block),
            ) as compile_mock,
        ):
            compiled = model_utils.compile_transformer(
                self.args(),
                transformer,
                [transformer.blocks],
                disable_linear=True,
                offloaders=[offloader],
            )
            compiled.blocks[1](torch.ones(1, 2))

        self.assertIs(originals[0], compiled.blocks[0])
        self.assertIs(originals[2], compiled.blocks[2])
        self.assertIsInstance(compiled.blocks[1], self.CompiledBlock)
        self.assertEqual(1, compile_mock.call_count)
        self.assertEqual([1], compiled._musubi_compile_state["plans"][0]["compile_indices"])
        self.assertEqual([0, 2], compiled._musubi_compile_state["plans"][0]["eager_indices"])
        self.assertEqual(1, compiled._musubi_compile_state["plans"][0]["compile_eligible_linears"])
        self.assertEqual(1, len(compiled._musubi_compile_state["verified"]))
        self.assertTrue(all(not hasattr(block[0], "_eager_forward") for block in originals))

    def test_zero_safe_blocks_leaves_transformer_eager(self):
        transformer = self.Transformer()
        originals = list(transformer.blocks)
        offloader = SimpleNamespace(compile_safe_block_indices=lambda: [])
        with (
            patch("musubi_tuner.utils.model_utils.ensure_training_compile_environment"),
            patch("musubi_tuner.utils.model_utils.torch.compile") as compile_mock,
        ):
            compiled = model_utils.compile_transformer(
                self.args(),
                transformer,
                [transformer.blocks],
                disable_linear=True,
                offloaders=[offloader],
            )

        self.assertEqual(originals, list(compiled.blocks))
        compile_mock.assert_not_called()
        self.assertFalse(compiled._musubi_compile_state["enabled"])
        self.assertEqual(0, compiled._musubi_compile_state["expected_compiled_blocks"])

    def test_missing_selector_falls_back_to_established_compile_policy(self):
        transformer = self.Transformer()
        with (
            patch("musubi_tuner.utils.model_utils.ensure_training_compile_environment"),
            patch(
                "musubi_tuner.utils.model_utils.torch.compile",
                side_effect=lambda block, **_kwargs: self.CompiledBlock(block),
            ) as compile_mock,
        ):
            compiled = model_utils.compile_transformer(
                self.args(),
                transformer,
                [transformer.blocks],
                disable_linear=True,
                offloaders=None,
            )

        self.assertEqual(3, compile_mock.call_count)
        self.assertEqual(
            "all_blocks_fallback_missing_residency_selector",
            compiled._musubi_compile_state["policy"],
        )
        self.assertIn("no residency selector", compiled._musubi_compile_state["policy_fallback_reason"])
        self.assertTrue(all(hasattr(block._orig_mod[0], "_eager_forward") for block in compiled.blocks))

    def test_every_training_compile_call_site_supplies_residency_selectors(self):
        source_root = Path(__file__).parents[1] / "src" / "musubi_tuner"
        missing = []
        call_count = 0
        for path in sorted(source_root.glob("*_train_network.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "compile_transformer"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "model_utils"
                ):
                    continue
                call_count += 1
                if not any(keyword.arg == "offloaders" for keyword in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")

        self.assertGreater(call_count, 0)
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
