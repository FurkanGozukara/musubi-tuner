<script>
	import FormField from '$lib/components/FormField.svelte';
	import FormSelect from '$lib/components/FormSelect.svelte';
	import FormToggle from '$lib/components/FormToggle.svelte';
	import FormGroup from '$lib/components/FormGroup.svelte';
	import PathInput from '$lib/components/PathInput.svelte';
	import { projectConfig, projectLoaded, updateSection } from '$lib/stores/project.js';
	import { advancedMode } from '$lib/stores/uiMode.js';

	function update(key, value) { updateSection('training', key, value); }

	let t = $derived($projectConfig?.training || {});

	const emptyModality = () => ({ is_generated: true, conditions: [] });
	const emptyRecipe = () => ({ enabled: false, per_sample_loss: 'auto', video: emptyModality(), audio: emptyModality() });

	// The recipe is stored structurally in training.conditioning_recipe; default-shape it for safety.
	let recipe = $derived.by(() => {
		const r = t.conditioning_recipe;
		if (!r || typeof r !== 'object') return emptyRecipe();
		return {
			enabled: !!r.enabled,
			per_sample_loss: r.per_sample_loss || 'auto',
			video: { is_generated: r.video?.is_generated !== false, conditions: r.video?.conditions || [] },
			audio: { is_generated: r.audio?.is_generated !== false, conditions: r.audio?.conditions || [] },
		};
	});

	function writeRecipe(next) { update('conditioning_recipe', next); }
	function clone(r) { return JSON.parse(JSON.stringify(r)); }

	const VIDEO_TYPES = ['first_frame', 'spatial_crop', 'inpaint', 'extend', 'reference'];
	const AUDIO_TYPES = ['extend', 'inpaint', 'reference'];
	const TYPE_LABEL = {
		first_frame: 'First frame', spatial_crop: 'Spatial crop', inpaint: 'Inpaint / outpaint',
		extend: 'Extend (prefix/suffix)', reference: 'Reference',
	};
	const TYPE_HELP = {
		first_frame: 'Keep frame 0 clean as conditioning (timestep 0, loss-excluded).',
		spatial_crop: 'Keep a rectangular region clean (region set per-dataset via spatial_crop_region). Outpaint the surround.',
		inpaint: "Use the dataset's loss-mask as a clean region: keep the masked area, generate the rest (invert = outpaint).",
		extend: 'Keep the first N / last N latent frames clean so the model extends a clip forward / backward.',
		reference: 'Concatenate a reference clip as conditioning (selects the IC-LoRA strategy automatically). Requires reference latents in the dataset.',
	};

	function newCondition(type) {
		// extend defaults prefix=1 so a freshly-added extend is valid (the trainer rejects prefix=suffix=0).
		return { type, probability: null, invert: false, threshold: 0.5, prefix: type === 'extend' ? 1 : 0, suffix: 0, prefix_p: null, suffix_p: null };
	}
	function availableTypes(modality) {
		const used = new Set((recipe[modality].conditions || []).map((c) => c.type));
		return (modality === 'video' ? VIDEO_TYPES : AUDIO_TYPES).filter((ty) => !used.has(ty));
	}
	function addCondition(modality, type) {
		const r = clone(recipe);
		r[modality].conditions = [...r[modality].conditions, newCondition(type)];
		r.enabled = true;
		writeRecipe(r);
	}
	function removeCondition(modality, idx) {
		const r = clone(recipe);
		r[modality].conditions = r[modality].conditions.filter((_, i) => i !== idx);
		writeRecipe(r);
	}
	function setCond(modality, idx, key, value) {
		const r = clone(recipe);
		r[modality].conditions[idx][key] = value;
		writeRecipe(r);
	}
	function setGenerated(modality, value) {
		const r = clone(recipe);
		r[modality].is_generated = value;
		// Freezing a modality is itself an activating action (directional training); without this a
		// freeze-only recipe stays inactive and silently trains plain joint.
		if (!value) r.enabled = true;
		writeRecipe(r);
	}
	function setPSL(value) { const r = clone(recipe); r.per_sample_loss = value; writeRecipe(r); }

	function numOrNull(v) { return v === '' || v === null || v === undefined ? null : Number(v); }

	// active = a real recipe (>=1 condition or a frozen modality). Mirrors the backend gate.
	let recipeActive = $derived(
		recipe.enabled && (recipe.video.conditions.length > 0 || recipe.audio.conditions.length > 0
			|| !recipe.video.is_generated || !recipe.audio.is_generated)
	);
	let hasVideoRef = $derived(recipe.video.conditions.some((c) => c.type === 'reference'));
	let hasAudioRef = $derived(recipe.audio.conditions.some((c) => c.type === 'reference'));
	let isAvIc = $derived(hasVideoRef && hasAudioRef);
	let audioModeOn = $derived(['av', 'audio'].includes(t.ltx2_mode));
	let videoModeOn = $derived(['av', 'video'].includes(t.ltx2_mode));

	// Live TOML preview — mirrors gui_dashboard/toml_export.py:render_conditioning_toml.
	function condLines(name, c) {
		const out = [`[[${name}.conditions]]`, `type = "${c.type}"`];
		const p = (k, v) => out.push(`${k} = ${v}`);
		if (c.type === 'extend') {
			p('prefix', Number(c.prefix || 0)); p('suffix', Number(c.suffix || 0));
			if (c.probability != null) p('probability', Number(c.probability));
			if (c.prefix_p != null) p('prefix_p', Number(c.prefix_p));
			if (c.suffix_p != null) p('suffix_p', Number(c.suffix_p));
		} else if (c.type === 'inpaint') {
			if (c.probability != null) p('probability', Number(c.probability));
			if (c.invert) p('invert', 'true');
			if (Number(c.threshold) !== 0.5) p('threshold', Number(c.threshold));
		} else if (c.type === 'spatial_crop') {
			if (c.probability != null) p('probability', Number(c.probability));
			if (c.invert) p('invert', 'true');
		} else if (c.probability != null && !(name === 'audio' && c.type === 'reference')) p('probability', Number(c.probability));
		out.push('');
		return out;
	}
	let previewToml = $derived.by(() => {
		const r = recipe;
		const lines = [];
		if (r.per_sample_loss === 'on') lines.push('per_sample_loss = true', '');
		else if (r.per_sample_loss === 'off') lines.push('per_sample_loss = false', '');
		for (const name of ['video', 'audio']) {
			const mod = r[name];
			if (mod.conditions.length === 0 && mod.is_generated) continue;
			lines.push(`[${name}]`);
			if (!mod.is_generated) lines.push('is_generated = false');
			lines.push('');
			for (const c of mod.conditions) lines.push(...condLines(name, c));
		}
		return lines.join('\n').trim() || '# (no conditions yet — add one above)';
	});

	// Validation: catch recipes the trainer will reject (or silently mis-train) BEFORE launch. The rules
	// mirror the trainer: v2a (frozen video) rejects spatial_crop/inpaint/extend (first_frame auto-disables);
	// a2v (frozen audio) allows audio conditions; a reference can't combine with a frozen modality; extend
	// needs prefix or suffix > 0 and no negative span; probabilities/threshold stay in [0, 1]; audio
	// conditions need an audio-bearing mode and video conditions need a video-bearing mode. Kept in sync
	// with the Python parity oracle in tests/test_ltx2_conditioning_correctness_harness.py (tags R1..R10).
	let recipeIssues = $derived.by(() => {
		const issues = [];
		if (!recipeActive) return issues;
		const r = recipe;
		const vFrozen = !r.video.is_generated;
		const aFrozen = !r.audio.is_generated;
		const anyRef = hasVideoRef || hasAudioRef;
		const vConflict = r.video.conditions.filter((c) => ['spatial_crop', 'inpaint', 'extend'].includes(c.type));
		if (vFrozen && aFrozen)
			issues.push({ level: 'error', msg: 'Both modalities are frozen — at least one must be Generated.' });
		if (vFrozen && vConflict.length)
			issues.push({ level: 'error', msg: 'Video is frozen (v2a) but has spatial_crop / inpaint / extend conditions, which it cannot combine with. Remove them, or set Video to Generate.' });
		if (anyRef && (vFrozen || aFrozen))
			issues.push({ level: 'error', msg: 'A reference condition cannot be combined with a frozen modality (directional). Remove the reference, or Generate both modalities.' });
		if ((vFrozen || aFrozen) && String(t.ltx2_mode) !== 'av')
			issues.push({ level: 'error', msg: 'Directional training (a frozen modality) requires Mode = av. Set Mode to av on the Training tab, or Generate both modalities.' });
		if (vFrozen && (t.video_anchor_training || t.hfato))
			issues.push({ level: 'error', msg: 'Video is frozen (v2a) but video-anchor training / HFATO is enabled — they modify the video the recipe keeps clean. Disable them.' });
		for (const [name, mod] of [['video', r.video], ['audio', r.audio]]) {
			for (const c of mod.conditions) {
				if (c.type === 'extend' && Number(c.prefix || 0) <= 0 && Number(c.suffix || 0) <= 0)
					issues.push({ level: 'error', msg: `The ${name} extend condition needs a prefix or suffix greater than 0.` });
				if (c.type === 'extend' && (Number(c.prefix || 0) < 0 || Number(c.suffix || 0) < 0))
					issues.push({ level: 'error', msg: `The ${name} extend prefix / suffix cannot be negative.` });
				if (c.probability === 0)
					issues.push({ level: 'warn', msg: `The ${name} ${c.type} condition has probability 0 — it will never apply.` });
				for (const key of ['probability', 'prefix_p', 'suffix_p']) {
					const val = c[key];
					if (val != null && (Number(val) < 0 || Number(val) > 1))
						issues.push({ level: 'error', msg: `The ${name} ${c.type} ${key} must be between 0 and 1.` });
				}
				if (c.type === 'inpaint' && c.threshold != null && (Number(c.threshold) < 0 || Number(c.threshold) > 1))
					issues.push({ level: 'error', msg: `The ${name} inpaint threshold must be between 0 and 1.` });
			}
		}
		if (!audioModeOn && (r.audio.conditions.length > 0 || aFrozen))
			issues.push({ level: 'error', msg: 'There are audio conditions but Mode is not av / audio — they fail at launch. Remove them, or set Mode to av / audio on the Training tab.' });
		if (!videoModeOn && r.video.conditions.length > 0)
			issues.push({ level: 'error', msg: 'There are video conditions but Mode is audio-only — they fail at launch. Remove them, or set Mode to av / video on the Training tab.' });
		if (!r.video.conditions.some((c) => c.type === 'first_frame') && !vFrozen && !aFrozen && !anyRef)
			issues.push({ level: 'warn', msg: 'This recipe turns OFF first-frame (I2V) conditioning that was on by default. Add a First frame condition to keep it.' });
		return issues;
	});
</script>

{#if !$projectLoaded}
	<div class="p-6 text-[12px]" style="color: var(--text-muted);">Open or create a project to configure conditioning.</div>
{:else}
	<div class="space-y-3">
		<div class="p-3 text-[12px] leading-relaxed" style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); color: var(--text-secondary);">
			<div class="text-[11px] font-semibold uppercase tracking-wider mb-1" style="color: var(--text-muted);">Conditioning</div>
			Compose the training scheme by adding <strong>conditions</strong> per modality. Each modality is either <strong>generated</strong> (the model learns to produce it) or <strong>frozen</strong> (used as clean conditioning — this is directional a2v / v2a training). Conditions compose; the trainer applies each at its per-sample probability. This builds a recipe automatically — no file to write.
			<span class="block mt-1" style="color: var(--text-muted);">Intrinsic conditions (first-frame, spatial-crop, inpaint, extend) keep a region clean (timestep 0) and exclude it from the loss. <strong>Reference</strong> selects the IC-LoRA strategy automatically. Per-sample loss normalization is enabled automatically whenever conditioning is active.</span>
		</div>

		{#if recipeIssues.length > 0}
			<div class="space-y-1">
				{#each recipeIssues as issue}
					<div class="flex items-start gap-2 px-2.5 py-1.5 text-[11px]" style="border-radius: var(--radius-sm); border: 1px solid {issue.level === 'error' ? 'var(--danger)' : issue.level === 'warn' ? 'var(--warning, var(--accent))' : 'var(--border-subtle)'}; background: {issue.level === 'error' ? 'var(--danger-muted, var(--bg-elevated))' : 'var(--bg-elevated)'}; color: {issue.level === 'error' ? 'var(--danger)' : 'var(--text-secondary)'};">
						<span class="flex-shrink-0 font-bold" style="color: {issue.level === 'error' ? 'var(--danger)' : issue.level === 'warn' ? 'var(--warning, var(--accent))' : 'var(--text-muted)'};">{issue.level === 'error' ? '!' : issue.level === 'warn' ? '!' : 'i'}</span>
						<span>{issue.msg}</span>
					</div>
				{/each}
			</div>
		{/if}

		{#each [['video', 'Video'], ['audio', 'Audio']] as [mod, label]}
			{#if mod === 'video' || audioModeOn}
				<FormGroup title={label}>
					<div class="space-y-2 pt-2">
						<div class="w-full sm:w-2/3">
							<FormToggle label="Generate this modality (off = freeze as conditioning)" checked={recipe[mod].is_generated} onchange={(e) => setGenerated(mod, e.target.checked)} tooltip="On = the model generates this modality. Off = freeze it as clean conditioning (directional a2v / v2a)." />
						</div>

						{#each recipe[mod].conditions as c, i}
							<div class="p-2 space-y-1.5" style="background: var(--bg-elevated); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);">
								<div class="flex items-center justify-between">
									<span class="text-[12px] font-semibold" style="color: var(--text-primary);">{TYPE_LABEL[c.type] || c.type}</span>
									<button type="button" onclick={() => removeCondition(mod, i)} class="px-1.5 text-[14px] leading-none" style="color: var(--text-muted);" aria-label="Remove condition"
										onmouseenter={(e) => e.currentTarget.style.color = 'var(--danger)'} onmouseleave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}>&times;</button>
								</div>
								<p class="text-[11px]" style="color: var(--text-muted);">{TYPE_HELP[c.type]}</p>
								{#if c.type === 'extend'}
									<div class="grid grid-cols-3 gap-2">
										<FormField type="number" label="Prefix frames" value={c.prefix ?? 0} oninput={(e) => setCond(mod, i, 'prefix', Number(e.target.value))} min={0} step="1" tooltip="Leading latent frames kept clean (forward extension). 0 = off." />
										<FormField type="number" label="Suffix frames" value={c.suffix ?? 0} oninput={(e) => setCond(mod, i, 'suffix', Number(e.target.value))} min={0} step="1" tooltip="Trailing latent frames kept clean (backward extension). 0 = off." />
										<FormField type="number" label="Probability" value={c.probability ?? ''} oninput={(e) => setCond(mod, i, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="1.0" tooltip="Per-sample probability of applying extension." />
									</div>
									{#if $advancedMode}
										<div class="grid grid-cols-2 gap-2">
											<FormField type="number" label="Prefix p" value={c.prefix_p ?? ''} oninput={(e) => setCond(mod, i, 'prefix_p', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="shared" tooltip="Independent prefix-side probability. Blank = use the shared probability." />
											<FormField type="number" label="Suffix p" value={c.suffix_p ?? ''} oninput={(e) => setCond(mod, i, 'suffix_p', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="shared" tooltip="Independent suffix-side probability. Blank = use the shared probability." />
										</div>
									{/if}
								{:else if c.type === 'inpaint'}
									<div class="grid grid-cols-3 gap-2">
										<FormField type="number" label="Probability" value={c.probability ?? ''} oninput={(e) => setCond(mod, i, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="0.0" tooltip="Per-sample probability of applying the mask as conditioning." />
										<FormField type="number" label="Threshold" value={c.threshold ?? 0.5} oninput={(e) => setCond(mod, i, 'threshold', e.target.value === '' ? 0.5 : Number(e.target.value))} min={0} max={1} step="0.05" tooltip="Binarization threshold (strict >). Mask values above this are kept clean." />
										<FormToggle label="Invert (outpaint)" checked={c.invert} onchange={(e) => setCond(mod, i, 'invert', e.target.checked)} tooltip="Condition the complement of the mask (generate the masked region)." />
									</div>
								{:else if c.type === 'spatial_crop'}
									<div class="grid grid-cols-2 gap-2">
										<FormField type="number" label="Probability" value={c.probability ?? ''} oninput={(e) => setCond(mod, i, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder="0.0" tooltip="Per-sample probability of applying the crop region as conditioning." />
										<FormToggle label="Invert (condition outside)" checked={c.invert} onchange={(e) => setCond(mod, i, 'invert', e.target.checked)} tooltip="Condition outside the rectangle instead of inside." />
									</div>
								{:else if mod === 'audio' && c.type === 'reference'}
									<p class="text-[11px]" style="color: var(--text-muted);">An audio reference takes no parameters — its presence selects the IC-LoRA strategy. Set the reference audio in the dataset config.</p>
								{:else}
									<div class="grid grid-cols-2 gap-2">
										<FormField type="number" label={c.type === 'reference' ? 'Keep probability' : 'Probability'} value={c.probability ?? ''} oninput={(e) => setCond(mod, i, 'probability', numOrNull(e.target.value))} min={0} max={1} step="0.05" placeholder={c.type === 'first_frame' ? '0.1' : c.type === 'reference' ? '1.0' : 'default'} tooltip={c.type === 'reference' ? 'Per-sample probability the reference is KEPT (values below 1.0 drop it for CFG-style training).' : 'Per-sample probability of applying first-frame conditioning.'} />
									</div>
								{/if}
							</div>
						{/each}

						{#if availableTypes(mod).length > 0}
							<div class="flex flex-wrap items-center gap-1.5 pt-1">
								<span class="text-[11px]" style="color: var(--text-muted);">Add condition:</span>
								{#each availableTypes(mod) as ty}
									<button type="button" onclick={() => addCondition(mod, ty)}
										class="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium"
										style="color: var(--accent); background: var(--accent-subtle-bg); border: 1px dashed var(--accent-muted); border-radius: var(--radius-full);"
										onmouseenter={(e) => e.currentTarget.style.borderColor = 'var(--accent)'} onmouseleave={(e) => e.currentTarget.style.borderColor = 'var(--accent-muted)'}>
										+ {TYPE_LABEL[ty] || ty}
									</button>
								{/each}
							</div>
						{:else}
							<p class="text-[11px]" style="color: var(--text-muted);">All condition types added.</p>
						{/if}
					</div>
				</FormGroup>
			{:else}
				<FormGroup title="Audio">
					<div class="pt-2">
						<p class="text-[11px]" style="color: var(--text-muted);">Audio conditions appear here when <strong>Mode</strong> is set to <code>av</code> or <code>audio</code> (Training tab). In video-only mode there is no audio stream to condition.</p>
					</div>
				</FormGroup>
			{/if}
		{/each}

		<FormGroup title="Loss & Preview">
			<div class="space-y-2 pt-2">
				<div class="w-full sm:w-2/3">
					<FormSelect label="Per-sample loss" value={recipe.per_sample_loss} options={[{ value: 'auto', label: 'Auto (enable when conditioning is active)' }, { value: 'on', label: 'On (force per-sample)' }, { value: 'off', label: 'Off (force batch-global)' }]} onchange={(e) => setPSL(e.target.value)} tooltip="Auto enables per-sample renormalization whenever conditioning is active (recommended). On/Off force it." />
				</div>
				<details class="text-[11px]">
					<summary class="cursor-pointer" style="color: var(--text-muted);">Recipe preview (TOML)</summary>
					<pre class="mt-1 p-2 overflow-x-auto" style="background: var(--bg-base); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); color: var(--text-secondary); white-space: pre;">{previewToml}</pre>
				</details>
			</div>
		</FormGroup>

		{#if hasAudioRef && $advancedMode}
			<FormGroup title="Reference fine-tuning (AV)">
				<div class="space-y-2 pt-2">
					<p class="text-[11px]" style="color: var(--text-muted);">Extra options for audio-reference / AV reference conditioning (compose with a reference condition above).</p>
					{#if isAvIc}
						<div class="grid grid-cols-2 gap-2">
							<FormSelect fieldPath="training.av_cross_attention_mode" value={t.av_cross_attention_mode || 'both'} options={[{ value: 'both', label: 'both' }, { value: 'a2v_only', label: 'a2v_only' }, { value: 'v2a_only', label: 'v2a_only' }, { value: 'none', label: 'none' }]} onchange={(e) => update('av_cross_attention_mode', e.target.value)} tooltip="AV cross-modal direction control." />
							<FormToggle fieldPath="training.av_multi_ref" checked={t.av_multi_ref ?? false} onchange={(e) => update('av_multi_ref', e.target.checked)} tooltip="Mark this AV run as multi-reference (uses the plural dataset reference fields)." />
						</div>
					{/if}
					<div class="grid grid-cols-3 gap-x-4 gap-y-1">
						<FormToggle fieldPath="training.audio_ref_use_negative_positions" checked={t.audio_ref_use_negative_positions ?? false} onchange={(e) => update('audio_ref_use_negative_positions', e.target.checked)} tooltip="Place reference-audio token positions in negative time." />
						<FormToggle fieldPath="training.audio_ref_mask_cross_attention_to_reference" checked={t.audio_ref_mask_cross_attention_to_reference ?? false} onchange={(e) => update('audio_ref_mask_cross_attention_to_reference', e.target.checked)} tooltip="Video attends only to target audio, not reference-audio tokens." />
						<FormToggle fieldPath="training.audio_ref_mask_reference_from_text_attention" checked={t.audio_ref_mask_reference_from_text_attention ?? false} onchange={(e) => update('audio_ref_mask_reference_from_text_attention', e.target.checked)} tooltip="Block reference-audio tokens from attending to text (ignored in AV IC modality-path mode)." />
					</div>
					<FormField type="number" fieldPath="training.audio_ref_identity_guidance_scale" value={t.audio_ref_identity_guidance_scale ?? 0.0} oninput={(e) => update('audio_ref_identity_guidance_scale', Number(e.target.value))} step="0.1" min={0} tooltip="Extra forward pass without reference to isolate speaker identity (0 = off, ~3.0 typical)." />
					<div class="grid grid-cols-3 items-end gap-x-4 gap-y-1">
						<FormToggle fieldPath="training.av_bimodal_cfg" checked={t.av_bimodal_cfg ?? false} onchange={(e) => update('av_bimodal_cfg', e.target.checked)} tooltip="Extra forward pass with cross-modal attention disabled to strengthen independent audio/video." />
						{#if t.av_bimodal_cfg}
							<FormField type="number" fieldPath="training.av_bimodal_scale" value={t.av_bimodal_scale ?? 3.0} oninput={(e) => update('av_bimodal_scale', Number(e.target.value))} step="0.1" min={1} tooltip="Bimodal guidance strength (scale-1)*delta. Default 3.0." />
						{/if}
					</div>
				</div>
			</FormGroup>
		{/if}

		<FormGroup title="Guidance (orthogonal)">
			<div class="space-y-2 pt-2">
				<p class="text-[11px]" style="color: var(--text-muted);">Latent-frame guidance, independent of the recipe above and composes with any conditioning.</p>
				<div class="pt-1 space-y-2">
					<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Endpoint Keyframe Training</span>
					<p class="text-[11px]" style="color: var(--text-muted);">Append first / last / random-interior latent frames of the target as clean keyframe tokens. No-ops while off.</p>
					<FormToggle fieldPath="training.keyframe_endpoint_training" checked={t.keyframe_endpoint_training ?? false} onchange={(e) => update('keyframe_endpoint_training', e.target.checked)} tooltip="Master enable for endpoint-keyframe training." />
					{#if t.keyframe_endpoint_training}
						<div class="grid grid-cols-2 gap-2">
							<FormField type="number" fieldPath="training.keyframe_first_frame_p" value={t.keyframe_first_frame_p ?? 1.0} oninput={(e) => update('keyframe_first_frame_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending the first latent frame at frame_idx=0." />
							<FormField type="number" fieldPath="training.keyframe_last_frame_p" value={t.keyframe_last_frame_p ?? 1.0} oninput={(e) => update('keyframe_last_frame_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending the last latent frame." />
							<FormField type="number" fieldPath="training.keyframe_random_interior_p" value={t.keyframe_random_interior_p ?? 0.0} oninput={(e) => update('keyframe_random_interior_p', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of appending random interior latent frames." />
							<FormField type="number" fieldPath="training.keyframe_max_random_interior" value={t.keyframe_max_random_interior ?? 0} oninput={(e) => update('keyframe_max_random_interior', Number(e.target.value))} step="1" min={0} tooltip="Cap on random interior keyframes per batch." />
						</div>
					{/if}
				</div>
				<div class="pt-3 space-y-2" style="border-top: 1px solid var(--border-subtle);">
					<span class="text-[11px] font-medium uppercase tracking-wider" style="color: var(--text-muted);">Video Anchor Training</span>
					<p class="text-[11px]" style="color: var(--text-muted);">Replace selected target frames in the noisy input with the clean target latent, zero their timesteps, exclude from loss.</p>
					<FormToggle fieldPath="training.video_anchor_training" checked={t.video_anchor_training ?? false} onchange={(e) => update('video_anchor_training', e.target.checked)} tooltip="Master enable for video-anchor training." />
					{#if t.video_anchor_training}
						<div class="grid grid-cols-3 gap-2">
							<FormField type="number" fieldPath="training.video_anchor_probability" value={t.video_anchor_probability ?? 0.5} oninput={(e) => update('video_anchor_probability', Number(e.target.value))} step="0.05" min={0} max={1} tooltip="Per-sample probability of applying anchor training." />
							<FormField type="number" fieldPath="training.video_anchor_count" value={t.video_anchor_count ?? 1} oninput={(e) => update('video_anchor_count', Number(e.target.value))} step="1" min={0} tooltip="Number of random anchors per sample when random anchors are enabled." />
							<FormSelect fieldPath="training.video_anchor_strategy" value={t.video_anchor_strategy || 'endpoints_random'} options={[{ value: 'endpoints', label: 'Endpoints' }, { value: 'random', label: 'Random' }, { value: 'endpoints_random', label: 'Endpoints + Random' }]} onchange={(e) => update('video_anchor_strategy', e.target.value)} tooltip="Anchor placement strategy." />
						</div>
					{/if}
				</div>
			</div>
		</FormGroup>

		{#if $advancedMode}
			<FormGroup title="External recipe file (advanced)">
				<div class="space-y-2 pt-2">
					<p class="text-[11px]" style="color: var(--text-muted);">Use an external TOML recipe file instead of the builder above. Used only when the builder is empty; the builder takes priority when it has conditions.</p>
					<PathInput fieldPath="training.ltx2_conditioning_config" value={t.ltx2_conditioning_config || ''} oninput={(e) => update('ltx2_conditioning_config', e.target.value)} showFiles tooltip="Path to a TOML conditioning recipe with [video]/[audio] blocks." />
				</div>
			</FormGroup>
		{/if}
	</div>
{/if}
