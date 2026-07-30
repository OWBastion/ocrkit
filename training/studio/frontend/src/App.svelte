<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte'

  type ReviewCounts = { total: number; accepted: number; pending: number; rejected: number }
  type Batch = { batch_id: string; sources: number; train_sources: number; holdout_sources: number; quality_warnings: number; layout_version: string; review?: ReviewCounts }
  type Row = { crop: string; roi: string; review_status: string; candidate_text?: string; transcription?: string; confidence?: number; rapidocr_text?: string; rapidocr_confidence?: number; vision_text?: string; vision_confidence?: number }
  type CropZoom = 'auto' | 1 | 2 | 3 | 4
  type TrainingState = {
    status?: string
    pid?: number
    log?: string
    log_tail?: string
    command?: string[]
  }

  let batches: Batch[] = []
  let batch: Batch | null = null
  let selected: Row | null = null
  let rows: Row[] = []
  let split = 'train'
  let status = 'pending'
  let files: File[] = []
  let holdoutRatio = 0.2
  let notice = ''
  let error = ''
  let busy = false
  let active = 'import'
  let training: TrainingState | null = null
  let trainingUpdatedAt = ''
  let trainingPolling = false
  let followLog = true
  let logEl: HTMLPreElement | null = null
  let cropZoom: CropZoom = 'auto'
  let cropNatural = { w: 0, h: 0 }
  let trainingPollTimer: ReturnType<typeof setInterval> | null = null
  const cropZoomSteps: CropZoom[] = ['auto', 1, 2, 3, 4]
  const TRAINING_POLL_MS = 2000
  const supportedClipboardTypes = new Set(['image/png', 'image/jpeg', 'image/webp'])

  const nav = [
    ['import', '1', '导入'],
    ['candidates', '2', '候选'],
    ['review', '3', '复核'],
    ['dataset', '4', '标签'],
    ['training', '5', '训练'],
  ] as const

  function openStep(step: (typeof nav)[number][0]) {
    if (step !== 'training') stopTrainingPoll()
    active = step
    if (step === 'review' && batch) void refreshReview()
    if (step === 'training' && batch) void loadTrainingStep()
  }

  async function selectBatch(batchId: string) {
    batch = batches.find((item) => item.batch_id === batchId) || null
    selected = null
    rows = []
    if (active === 'training') {
      if (batch) await loadTrainingStep()
      else {
        training = null
        stopTrainingPoll()
      }
    }
  }

  function batchHint() {
    if (!batch) return '请先导入截图并创建批次'
    const review = batch.review
    if (!review || review.total === 0) return '下一步：生成候选'
    if (review.pending > 0) return `下一步：复核剩余 ${review.pending} 条`
    return '下一步：生成标签，然后可启动训练'
  }

  async function request<T>(url: string, init?: RequestInit): Promise<T> {
    const response = await fetch(url, init)
    if (!response.ok) {
      const body = await response.json().catch(() => ({})) as { detail?: string }
      throw new Error(body.detail || `请求失败 (${response.status})`)
    }
    return response.json() as Promise<T>
  }

  function message(value: string, isError = false) {
    error = isError ? value : ''
    notice = isError ? '' : value
  }

  async function refreshBatches(selectId?: string) {
    batches = await request<Batch[]>('/api/batches')
    batch = batches.find((item) => item.batch_id === (selectId || batch?.batch_id)) || batches[0] || null
  }

  async function refreshReview(options?: { keepSelection?: boolean; preferCrops?: string[] }) {
    if (!batch) {
      rows = []
      selected = null
      return
    }
    const previousCrop = selected?.crop
    const data = await request<{ rows: Row[]; counts: ReviewCounts }>(`/api/batches/${batch.batch_id}/review?split=${split}&status=${status}`)
    rows = data.rows
    batch = { ...batch, review: data.counts }

    if (options?.preferCrops?.length) {
      const next = options.preferCrops
        .map((crop) => rows.find((row) => row.crop === crop))
        .find((row): row is Row => Boolean(row))
      if (next) {
        selectCandidate(next)
        return
      }
    }

    if (options?.keepSelection && previousCrop) {
      const stillHere = rows.find((row) => row.crop === previousCrop)
      if (stillHere) {
        selectCandidate(stillHere)
        return
      }
    }

    selected = null
  }

  async function importFiles() {
    if (!files.length) return message('先选择至少一张截图。', true)
    busy = true
    try {
      const data = new FormData()
      data.append('holdout_ratio', String(holdoutRatio))
      files.forEach((file) => data.append('files', file))
      const result = await request<{ batch: Batch }>('/api/batches', { method: 'POST', body: data })
      await refreshBatches(result.batch.batch_id)
      files = []
      message('批次已创建，进入候选步骤。')
      active = 'candidates'
    } catch (cause) { message(cause instanceof Error ? cause.message : '导入失败', true) } finally { busy = false }
  }

  function appendFiles(incoming: File[]) {
    files = [...files, ...incoming]
  }

  function clipboardName(file: File, timestamp: number, index: number) {
    const extension = file.type === 'image/jpeg' ? 'jpg' : file.type === 'image/webp' ? 'webp' : 'png'
    return `clipboard-${timestamp}-${index + 1}.${extension}`
  }

  function pasteImages(event: ClipboardEvent) {
    if (active !== 'import' || busy) return
    const images = Array.from(event.clipboardData?.items || [])
      .filter((item) => item.kind === 'file' && supportedClipboardTypes.has(item.type))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null)
    if (!images.length) return

    event.preventDefault()
    const timestamp = Date.now()
    appendFiles(images.map((file, index) => new File([file], clipboardName(file, timestamp, index), { type: file.type })))
    message(`已从剪贴板加入 ${images.length} 张截图。`)
  }

  async function candidates() {
    if (!batch) return message('先选择或创建批次。', true)
    busy = true
    try {
      const result = await request<{ review: ReviewCounts; summary: { reused_existing_candidates?: boolean } }>(`/api/batches/${batch.batch_id}/candidates`, { method: 'POST' })
      batch = { ...batch, review: result.review }
      message(result.summary.reused_existing_candidates ? '已打开现有候选，进入复核。' : '候选已生成，进入复核。')
      active = 'review'
      await refreshReview()
      if (rows[0]) selectCandidate(rows[0])
    } catch (cause) { message(cause instanceof Error ? cause.message : '候选生成失败', true) } finally { busy = false }
  }

  function confidenceLabel(value?: number | null) {
    if (value == null || Number.isNaN(value)) return '—'
    return `${Math.round(value * 100)}%`
  }

  function selectCandidate(row: Row) {
    selected = { ...row, transcription: row.transcription || row.candidate_text || '' }
    cropZoom = 'auto'
    cropNatural = { w: 0, h: 0 }
  }

  function onCropLoad(event: Event) {
    const image = event.currentTarget as HTMLImageElement
    cropNatural = { w: image.naturalWidth, h: image.naturalHeight }
  }

  function cropIsWideStrip() {
    return cropNatural.w > 0 && cropNatural.h > 0 && cropNatural.w / cropNatural.h >= 2.4
  }

  function cropDisplayScale() {
    if (!cropNatural.h) return 1
    if (cropZoom !== 'auto') return cropZoom
    // Short OCR strips need height-first upscale so glyphs stay legible.
    if (cropIsWideStrip() || cropNatural.h < 64) {
      const targetHeight = Math.min(140, Math.max(88, cropNatural.h * 3))
      return Math.min(8, Math.max(2, targetHeight / cropNatural.h))
    }
    if (cropNatural.h < 120) return Math.min(3, 120 / cropNatural.h)
    return 1
  }

  function cropImageStyle() {
    if (!cropNatural.w || !cropNatural.h) return 'max-width: 100%; height: auto;'
    const scale = cropDisplayScale()
    const width = Math.round(cropNatural.w * scale)
    const height = Math.round(cropNatural.h * scale)
    if (cropZoom === 'auto') {
      return `width: min(100%, ${width}px); height: auto; aspect-ratio: ${cropNatural.w} / ${cropNatural.h};`
    }
    return `width: ${width}px; height: ${height}px; max-width: none;`
  }

  function cropZoomLabel(step: CropZoom) {
    return step === 'auto' ? '自适应' : `${step}×`
  }

  function applyEngineText(text?: string | null) {
    if (!selected || !text) return
    selected = { ...selected, transcription: text }
  }

  function engineSelected(text?: string | null) {
    return Boolean(selected?.transcription && text && selected.transcription === text)
  }

  async function save(statusValue: 'accepted' | 'rejected') {
    if (!batch || !selected) return
    busy = true
    try {
      const currentIndex = rows.findIndex((row) => row.crop === selected?.crop)
      // Prefer the items after the current one so review advances forward.
      const followingCrops = currentIndex >= 0
        ? rows.slice(currentIndex + 1).map((row) => row.crop)
        : []
      const result = await request<{ row: Row; counts: ReviewCounts }>(`/api/batches/${batch.batch_id}/review`, {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          split,
          crop: selected.crop,
          status: statusValue,
          transcription: selected.transcription || selected.candidate_text || '',
        }),
      })
      batch = { ...batch, review: result.counts }
      await refreshReview({ preferCrops: followingCrops })
      if (selected) {
        message(statusValue === 'accepted' ? '已接受，已切换到下一条。' : '已拒绝，已切换到下一条。')
      } else if ((batch.review?.pending ?? 0) > 0 && status === 'pending') {
        message(statusValue === 'accepted' ? '已接受。当前筛选已到末尾。' : '已拒绝。当前筛选已到末尾。')
      } else if ((batch.review?.pending ?? 0) === 0) {
        message(statusValue === 'accepted' ? '已接受。待复核已全部处理完。' : '已拒绝。待复核已全部处理完。')
      } else {
        message(statusValue === 'accepted' ? '已接受。' : '已拒绝。')
      }
    } catch (cause) {
      message(cause instanceof Error ? cause.message : '保存失败', true)
    } finally {
      busy = false
    }
  }

  async function finalize() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try {
      const result = await request<Record<string, number>>(`/api/batches/${batch.batch_id}/finalize`, { method: 'POST' })
      message(`标签已生成：train ${result.validated_train}，holdout ${result.validated_holdout}。`)
      active = 'training'
      await loadTrainingStep()
    } catch (cause) { message(cause instanceof Error ? cause.message : '生成标签失败', true) } finally { busy = false }
  }

  function trainingStatusLabel(value?: string) {
    switch (value) {
      case 'training': return '运行中'
      case 'completed': return '成功'
      case 'failed': return '失败'
      case 'completed_or_failed': return '已结束'
      case 'not_started': return '未开始'
      default: return value || '未知'
    }
  }

  function trainingIsRunning(value?: string) {
    return value === 'training'
  }

  function trainingIsDone(value?: string) {
    return value === 'completed' || value === 'failed' || value === 'completed_or_failed'
  }

  function stopTrainingPoll() {
    if (trainingPollTimer) {
      clearInterval(trainingPollTimer)
      trainingPollTimer = null
    }
    trainingPolling = false
  }

  function startTrainingPoll() {
    if (trainingPollTimer || active !== 'training' || !batch) return
    trainingPolling = true
    trainingPollTimer = setInterval(() => {
      void refreshTraining({ silent: true })
    }, TRAINING_POLL_MS)
  }

  async function scrollLogToBottom() {
    if (!followLog || !logEl) return
    await tick()
    logEl.scrollTop = logEl.scrollHeight
  }

  async function refreshTraining(options?: { silent?: boolean }) {
    if (!batch) return
    try {
      const previousStatus = training?.status
      training = await request<TrainingState>(`/api/batches/${batch.batch_id}/training`)
      trainingUpdatedAt = new Date().toLocaleTimeString()
      if (trainingIsRunning(training.status)) {
        if (active === 'training') startTrainingPoll()
      } else {
        stopTrainingPoll()
        if (!options?.silent && previousStatus === 'training' && trainingIsDone(training.status)) {
          message(training.status === 'completed' ? 'Smoke 训练已成功结束。' : 'Smoke 训练已结束，请查看日志。')
        }
      }
      await scrollLogToBottom()
    } catch (cause) {
      stopTrainingPoll()
      if (!options?.silent) message(cause instanceof Error ? cause.message : '获取训练状态失败', true)
    }
  }

  async function loadTrainingStep() {
    await refreshTraining()
    if (trainingIsRunning(training?.status)) startTrainingPoll()
  }

  async function startTraining() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try {
      await request<TrainingState>(`/api/batches/${batch.batch_id}/training/smoke`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({}),
      })
      message('Smoke 训练已启动，状态将自动刷新。')
      followLog = true
      await refreshTraining()
      startTrainingPoll()
    } catch (cause) { message(cause instanceof Error ? cause.message : '训练启动失败', true) } finally { busy = false }
  }

  onMount(async () => { try { await refreshBatches() } catch { message('无法连接 Studio API。', true) } })
  onDestroy(stopTrainingPoll)
</script>

<svelte:window on:paste={pasteImages} />

<svelte:head><meta name="description" content="OCRKit Studio" /></svelte:head>

<main class="app-shell">
  <div class="app-frame">
    <header class="app-header">
      <div class="brand">
        <h1>OCRKit Studio</h1>
        <p class="flow-hint">{batchHint()}</p>
      </div>
      <label class="batch-picker">
        <span>批次</span>
        <select
          value={batch?.batch_id || ''}
          on:change={(event) => void selectBatch((event.currentTarget as HTMLSelectElement).value)}
        >
          <option value="">未选择</option>
          {#each batches as item}
            <option value={item.batch_id}>{item.batch_id}</option>
          {/each}
        </select>
      </label>
    </header>

    <nav class="step-nav" aria-label="操作流程">
      {#each nav as item}
        <button
          type="button"
          class:nav-current={active === item[0]}
          class="nav-item"
          on:click={() => openStep(item[0])}
        >
          <span>{item[1]}</span>{item[2]}
        </button>
      {/each}
    </nav>

    {#if notice || error}
      <div class:error-note={error} class="notice" role="status">{error || notice}</div>
    {/if}

    {#if active === 'import'}
      <section class="panel-grid">
        <div class="panel">
          <header class="panel-head">
            <h2>1. 导入截图</h2>
            <p>选择或粘贴 PNG / JPEG / WebP，设置 holdout 比例后创建批次。重复截图会按内容去重。</p>
          </header>
          <label class="file-pick">
            <input
              type="file"
              multiple
              accept="image/png,image/jpeg,image/webp"
              on:change={(event) => appendFiles(Array.from((event.currentTarget as HTMLInputElement).files || []))}
            />
            <span>{files.length ? `已选 ${files.length} 张` : '选择截图，或 ⌘V / Ctrl+V 粘贴'}</span>
          </label>
          <div class="panel-actions">
            <label class="field">
              Holdout 比例
              <input type="number" min="0" max="0.5" step="0.05" bind:value={holdoutRatio} />
            </label>
            <button class="button-primary" disabled={busy || !files.length} on:click={importFiles}>
              {busy ? '创建中…' : '创建批次'}
            </button>
          </div>
        </div>
        <aside class="panel panel-side">
          <h3>操作流程</h3>
          <ol class="flow-list">
            <li class:flow-current={active === 'import'}><b>导入</b>创建批次</li>
            <li><b>候选</b> 跑 RapidOCR + Vision</li>
            <li><b>复核</b> 接受 / 拒绝每条切片</li>
            <li><b>标签</b> 生成 train / holdout labels</li>
            <li><b>训练</b> 启动 CPU Smoke</li>
          </ol>
          {#if batch}
            <dl class="batch-stats">
              <div><dt>截图</dt><dd>{batch.sources}</dd></div>
              <div><dt>Train</dt><dd>{batch.train_sources}</dd></div>
              <div><dt>Holdout</dt><dd>{batch.holdout_sources}</dd></div>
              <div><dt>待复核</dt><dd>{batch.review?.pending ?? '—'}</dd></div>
            </dl>
          {/if}
        </aside>
      </section>
    {:else if active === 'candidates'}
      <section class="panel stage-panel">
        <header class="panel-head">
          <h2>2. 生成候选</h2>
          <p>对当前批次按固定 ROI 切片，并用 RapidOCR 与 Vision 写复核清单。若已生成过，会直接打开现有结果，不会覆盖人工修改。</p>
        </header>
        <div class="panel-actions">
          <button class="button-primary" disabled={!batch || busy} on:click={candidates}>
            {busy ? '生成中…' : batch?.review?.total ? '打开候选' : '生成候选'}
          </button>
        </div>
      </section>
    {:else if active === 'review'}
      <section class="review-layout">
        <aside class="review-list">
          <div class="filters">
            <select bind:value={split} on:change={() => refreshReview()}>
              <option value="train">Train</option>
              <option value="holdout">Holdout</option>
            </select>
            <select bind:value={status} on:change={() => refreshReview()}>
              <option value="pending">待复核</option>
              <option value="accepted">已接受</option>
              <option value="rejected">已拒绝</option>
              <option value="all">全部</option>
            </select>
            <button class="button-secondary" on:click={() => refreshReview()}>刷新</button>
          </div>
          <div class="countline">
            <b>{batch?.review?.pending || 0}</b> 待复核 / {batch?.review?.total || 0} 总计
          </div>
          {#if rows.length}
            {#each rows as row}
              <button
                type="button"
                class:selected-row={selected?.crop === row.crop}
                class="candidate"
                on:click={() => selectCandidate(row)}
              >
                <span>{row.roi}</span>
                <strong>{row.candidate_text || '无候选文本'}</strong>
                <small>{confidenceLabel(row.confidence)}</small>
              </button>
            {/each}
          {:else}
            <p class="empty">当前筛选无结果。若尚未生成候选，请回到上一步。</p>
          {/if}
        </aside>
        <article class="review-detail">
          {#if selected && batch}
            <header class="detail-header">
              <div>
                <p class="eyebrow">{selected.roi} · {selected.review_status}</p>
                <h2>3. 复核</h2>
              </div>
              {#if cropNatural.w}
                <p class="crop-meta" aria-live="polite">
                  {cropNatural.w}×{cropNatural.h}px · {cropZoom === 'auto' ? `自适应 ${cropDisplayScale().toFixed(1)}×` : `${cropZoom}×`}
                </p>
              {/if}
            </header>

            <section class="crop-panel" aria-label="切片预览">
              <div class="crop-toolbar">
                <span class="crop-toolbar-label">切片</span>
                <div class="crop-zoom" role="group" aria-label="切片缩放">
                  {#each cropZoomSteps as step}
                    <button
                      type="button"
                      class="crop-zoom-btn"
                      class:crop-zoom-active={cropZoom === step}
                      aria-pressed={cropZoom === step}
                      on:click={() => cropZoom = step}
                    >{cropZoomLabel(step)}</button>
                  {/each}
                </div>
              </div>
              <div class="crop-viewport" class:crop-viewport-scroll={cropZoom !== 'auto'}>
                <img
                  src={`/api/batches/${batch.batch_id}/crop?split=${split}&crop=${encodeURIComponent(selected.crop)}`}
                  alt="当前 OCR 切片"
                  class="crop-image"
                  class:crop-image-crisp={cropDisplayScale() > 1.25}
                  class:crop-image-wide={cropIsWideStrip()}
                  style={cropImageStyle()}
                  on:load={onCropLoad}
                />
              </div>
            </section>

            <div class="detail-copy">
              <p class="engine-hint">点选引擎结果填入转写，或直接编辑后接受 / 拒绝。</p>
              <div class="engine-picks" role="group" aria-label="双引擎识别结果">
                <button
                  type="button"
                  class="engine-pick"
                  class:engine-pick-active={engineSelected(selected.rapidocr_text)}
                  class:engine-pick-empty={!selected.rapidocr_text}
                  disabled={!selected.rapidocr_text}
                  on:click={() => applyEngineText(selected?.rapidocr_text)}
                >
                  <span class="engine-pick-head">
                    <span class="engine-name">RapidOCR</span>
                    <span
                      class="engine-conf"
                      class:engine-conf-high={(selected.rapidocr_confidence ?? 0) >= 0.98}
                      class:engine-conf-mid={(selected.rapidocr_confidence ?? 0) >= 0.9 && (selected.rapidocr_confidence ?? 0) < 0.98}
                    >{confidenceLabel(selected.rapidocr_confidence)}</span>
                  </span>
                  <strong class="engine-text">{selected.rapidocr_text || '无识别结果'}</strong>
                  {#if engineSelected(selected.rapidocr_text)}<span class="engine-chosen">已选用</span>{/if}
                </button>
                <button
                  type="button"
                  class="engine-pick"
                  class:engine-pick-active={engineSelected(selected.vision_text)}
                  class:engine-pick-empty={!selected.vision_text}
                  disabled={!selected.vision_text}
                  on:click={() => applyEngineText(selected?.vision_text)}
                >
                  <span class="engine-pick-head">
                    <span class="engine-name">Vision</span>
                    <span
                      class="engine-conf"
                      class:engine-conf-high={(selected.vision_confidence ?? 0) >= 0.98}
                      class:engine-conf-mid={(selected.vision_confidence ?? 0) >= 0.9 && (selected.vision_confidence ?? 0) < 0.98}
                    >{confidenceLabel(selected.vision_confidence)}</span>
                  </span>
                  <strong class="engine-text">{selected.vision_text || '无识别结果'}</strong>
                  {#if engineSelected(selected.vision_text)}<span class="engine-chosen">已选用</span>{/if}
                </button>
              </div>
              {#if selected.rapidocr_text && selected.vision_text && selected.rapidocr_text !== selected.vision_text}
                <p class="engine-disagree">两引擎不一致，请对照切片选择或手改。</p>
              {:else if selected.rapidocr_text && selected.vision_text && selected.rapidocr_text === selected.vision_text}
                <p class="engine-agree">两引擎一致。</p>
              {/if}
              <label class="transcription-field">
                <span>转写</span>
                <textarea bind:value={selected.transcription} aria-label="转写" rows="3"></textarea>
              </label>
              <div class="actions">
                <button class="button-secondary" disabled={busy} on:click={() => save('rejected')}>拒绝</button>
                <button class="button-primary" disabled={busy} on:click={() => save('accepted')}>接受</button>
              </div>
            </div>
          {:else}
            <div class="empty-detail">
              <h2>从左侧选择一条候选</h2>
              <p>筛选 train / holdout 与状态，点条目后对照切片校正转写。</p>
            </div>
          {/if}
        </article>
      </section>
    {:else if active === 'dataset'}
      <section class="panel stage-panel">
        <header class="panel-head">
          <h2>4. 生成标签</h2>
          <p>全部候选需已接受或拒绝。通过后写出 train / holdout recognition labels，并做格式校验。</p>
        </header>
        <div class="panel-actions">
          <button class="button-primary" disabled={!batch || busy} on:click={finalize}>
            {busy ? '生成中…' : '生成 labels'}
          </button>
        </div>
      </section>
    {:else}
      <section class="training-panel">
        <div class="training-head">
          <header class="panel-head">
            <h2>5. Smoke 训练</h2>
            <p>在当前批次 labels 上启动本地 CPU Smoke。运行中会自动刷新状态与日志。</p>
          </header>
          <div class="panel-actions">
            <button class="button-secondary" disabled={!batch || busy} on:click={() => refreshTraining()}>立即刷新</button>
            <button class="button-primary" disabled={!batch || busy || trainingIsRunning(training?.status)} on:click={startTraining}>
              {busy ? '启动中…' : trainingIsRunning(training?.status) ? '训练中…' : '启动训练'}
            </button>
          </div>
        </div>

        {#if !batch}
          <p class="empty">请先选择批次。</p>
        {:else if !training || training.status === 'not_started'}
          <p class="empty">尚未启动训练。生成 labels 后点击「启动训练」。</p>
        {:else}
          <div class="training-status" aria-live="polite">
            <span
              class="status-pill"
              class:status-running={trainingIsRunning(training.status)}
              class:status-done={training.status === 'completed'}
              class:status-failed={training.status === 'failed' || training.status === 'completed_or_failed'}
            >{trainingStatusLabel(training.status)}</span>
            {#if training.pid}<span class="status-meta">PID {training.pid}</span>{/if}
            {#if trainingPolling}<span class="status-meta status-live">自动刷新中 · 每 {TRAINING_POLL_MS / 1000}s</span>{/if}
            {#if trainingUpdatedAt}<span class="status-meta">更新于 {trainingUpdatedAt}</span>{/if}
          </div>

          {#if training.command?.length}
            <p class="training-command" title={training.command.join(' ')}>
              <span>命令</span>
              <code>{training.command.join(' ')}</code>
            </p>
          {/if}

          <section class="log-panel" aria-label="训练日志">
            <header class="log-toolbar">
              <span>日志</span>
              <div class="log-toolbar-actions">
                <label class="log-follow">
                  <input type="checkbox" bind:checked={followLog} on:change={() => void scrollLogToBottom()} />
                  跟随底部
                </label>
                {#if training.log}
                  <span class="status-meta log-path" title={training.log}>{training.log.split('/').slice(-3).join('/')}</span>
                {/if}
              </div>
            </header>
            <pre class="log-tail" bind:this={logEl}>{training.log_tail?.trimEnd() || '（暂无输出，启动后将显示在这里）'}</pre>
          </section>
        {/if}
      </section>
    {/if}
  </div>
</main>
