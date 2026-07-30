<script lang="ts">
  import { onMount } from 'svelte'

  type ReviewCounts = { total: number; accepted: number; pending: number; rejected: number }
  type Batch = { batch_id: string; sources: number; train_sources: number; holdout_sources: number; quality_warnings: number; layout_version: string; review: ReviewCounts }
  type Row = { crop: string; roi: string; review_status: string; candidate_text?: string; transcription?: string; confidence?: number; rapidocr_text?: string; rapidocr_confidence?: number; vision_text?: string; vision_confidence?: number }

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
  let training: Record<string, unknown> | null = null
  const supportedClipboardTypes = new Set(['image/png', 'image/jpeg', 'image/webp'])

  const nav = [
    ['import', '01', '导入'], ['candidates', '02', '候选'], ['review', '03', '复核'], ['dataset', '04', '数据集'], ['training', '05', '训练']
  ]

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

  async function refreshReview() {
    selected = null
    if (!batch) { rows = []; return }
    const data = await request<{ rows: Row[]; counts: ReviewCounts }>(`/api/batches/${batch.batch_id}/review?split=${split}&status=${status}`)
    rows = data.rows
    batch = { ...batch, review: data.counts }
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
      message('批次已创建。下一步生成候选。')
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
      message(result.summary.reused_existing_candidates ? '已保留并重新打开已有候选。' : '候选已生成；低置信度项需要人工复核。')
      active = 'review'
      await refreshReview()
    } catch (cause) { message(cause instanceof Error ? cause.message : '候选生成失败', true) } finally { busy = false }
  }

  async function save(statusValue: 'accepted' | 'rejected') {
    if (!batch || !selected) return
    busy = true
    try {
      const result = await request<{ row: Row; counts: ReviewCounts }>(`/api/batches/${batch.batch_id}/review`, {
        method: 'PUT', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ split, crop: selected.crop, status: statusValue, transcription: selected.transcription || selected.candidate_text || '' })
      })
      batch = { ...batch, review: result.counts }
      message(statusValue === 'accepted' ? '转写已接受并保存。' : '候选已拒绝。')
      await refreshReview()
    } catch (cause) { message(cause instanceof Error ? cause.message : '保存失败', true) } finally { busy = false }
  }

  async function finalize() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try { const result = await request<Record<string, number>>(`/api/batches/${batch.batch_id}/finalize`, { method: 'POST' }); message(`数据集已验证：train ${result.validated_train}，holdout ${result.validated_holdout}。`) }
    catch (cause) { message(cause instanceof Error ? cause.message : '数据集验证失败', true) } finally { busy = false }
  }

  async function startTraining() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try { training = await request(`/api/batches/${batch.batch_id}/training/smoke`, { method: 'POST' }); message('CPU Smoke 训练已在独立进程启动。'); await refreshTraining() }
    catch (cause) { message(cause instanceof Error ? cause.message : '训练启动失败', true) } finally { busy = false }
  }

  async function refreshTraining() { if (batch) training = await request(`/api/batches/${batch.batch_id}/training`) }

  onMount(async () => { try { await refreshBatches() } catch (cause) { message('无法连接本机 Studio API。', true) } })
</script>

<svelte:window on:paste={pasteImages} />

<svelte:head><meta name="description" content="OCRKit local dataset and training studio" /></svelte:head>

<main class="min-h-screen px-4 py-5 sm:px-8 lg:px-12">
  <section class="mx-auto max-w-7xl">
    <header class="grid gap-6 border-b border-ink/12 pb-7 lg:grid-cols-[1fr_auto] lg:items-end">
      <div>
        <p class="mb-3 text-xs font-semibold tracking-[.22em] text-clay">LOCAL / OFFLINE / REVIEWED</p>
        <h1 class="max-w-3xl text-5xl font-semibold tracking-[-.065em] text-ink sm:text-7xl">OCRKit<br /><i class="font-normal">Studio</i></h1>
        <p class="mt-5 max-w-xl text-base leading-7 text-ink/65">把截图变成可追溯训练数据。每一步都在本机批次工作区中完成。</p>
      </div>
      <label class="batch-picker"><span>当前批次</span><select value={batch?.batch_id || ''} on:change={(event) => { batch = batches.find((item) => item.batch_id === (event.currentTarget as HTMLSelectElement).value) || null; selected = null; rows = [] }}><option value="">尚未选择</option>{#each batches as item}<option value={item.batch_id}>{item.batch_id}</option>{/each}</select></label>
    </header>

    <nav class="my-5 flex gap-1 overflow-x-auto" aria-label="Studio 流程">
      {#each nav as item}
        <button class:nav-current={active === item[0]} class="nav-item" on:click={() => active = item[0]}><span>{item[1]}</span>{item[2]}</button>
      {/each}
    </nav>

    {#if notice || error}<div class:error-note={error} class="notice" role="status">{error || notice}</div>{/if}

    {#if active === 'import'}
      <section class="grid gap-8 lg:grid-cols-[1.2fr_.8fr]">
        <div class="drop-zone">
          <p class="eyebrow">新建批次</p><h2>导入原始截图</h2><p>支持 PNG、JPEG、WebP。系统会验证图像、计算 SHA-256 去重，并按整张截图分配 train / holdout。</p>
          <label class="file-pick"><input type="file" multiple accept="image/png,image/jpeg,image/webp" on:change={(event) => appendFiles(Array.from((event.currentTarget as HTMLInputElement).files || []))} /><span>{files.length ? `已选 ${files.length} 张截图` : '选择截图'}</span></label>
          <p class="mt-4 text-sm leading-6 text-ink/65">也可在此页面直接按 <kbd>⌘V</kbd>（Windows/Linux：<kbd>Ctrl+V</kbd>）粘贴平台复制的截图；可连续粘贴多张。</p>
          <div class="mt-7 flex flex-wrap items-end gap-4"><label class="field">Holdout 比例 <input type="number" min="0" max="0.5" step="0.05" bind:value={holdoutRatio} /></label><button class="button-primary" disabled={busy} on:click={importFiles}>{busy ? '处理中…' : '创建本地批次'}</button></div>
        </div>
        <aside class="side-note"><p class="eyebrow">隐私边界</p><p>原图、ROI、切片和日志只写入 <code>training/.work/studio/</code>，不会进入 Git、生产镜像或模型发布流程。</p></aside>
      </section>
    {:else if active === 'candidates'}
      <section class="stage"><div><p class="eyebrow">固定布局 / 双引擎</p><h2>生成可审核候选</h2><p>RapidOCR 与 macOS Vision 仅在文本一致且置信度均达到阈值时自动接受。再次执行会复用现有审核清单，不会覆盖人工修改。</p></div><button class="button-primary" disabled={!batch || busy} on:click={candidates}>{busy ? '生成中…' : '生成或打开候选'}</button></section>
    {:else if active === 'review'}
      <section class="review-layout">
        <aside class="review-list"><div class="filters"><select bind:value={split} on:change={refreshReview}><option value="train">Train</option><option value="holdout">Holdout</option></select><select bind:value={status} on:change={refreshReview}><option value="pending">待复核</option><option value="accepted">已接受</option><option value="rejected">已拒绝</option><option value="all">全部</option></select><button class="button-secondary" on:click={refreshReview}>刷新</button></div>
          <div class="countline"><b>{batch?.review.pending || 0}</b> 待复核 / {batch?.review.total || 0} 总计</div>
          {#if rows.length}{#each rows as row}<button class:selected-row={selected?.crop === row.crop} class="candidate" on:click={() => selected = { ...row, transcription: row.transcription || row.candidate_text || '' }}><span>{row.roi}</span><strong>{row.candidate_text || '无候选文本'}</strong><small>{Math.round((row.confidence || 0) * 100)}%</small></button>{/each}{:else}<p class="empty">此筛选下没有候选。生成候选后点击刷新。</p>{/if}
        </aside>
        <article class="review-detail">
          {#if selected && batch}<img src={`/api/batches/${batch.batch_id}/crop?split=${split}&crop=${encodeURIComponent(selected.crop)}`} alt="当前 OCR 切片" /><div class="detail-copy"><p class="eyebrow">{selected.roi} / {selected.review_status}</p><h2>校正转写</h2><textarea bind:value={selected.transcription} aria-label="人工转写"></textarea><dl><div><dt>RapidOCR</dt><dd>{selected.rapidocr_text || '—'} · {selected.rapidocr_confidence ?? '—'}</dd></div><div><dt>Vision</dt><dd>{selected.vision_text || '—'} · {selected.vision_confidence ?? '—'}</dd></div></dl><div class="actions"><button class="button-secondary" disabled={busy} on:click={() => save('rejected')}>拒绝</button><button class="button-primary" disabled={busy} on:click={() => save('accepted')}>接受转写</button></div></div>{:else}<div class="empty-detail"><p class="eyebrow">人工复核</p><h2>选择一条候选开始校正</h2><p>先在左侧选择数据分组和状态；键入准确文字后再接受。</p></div>{/if}
        </article>
      </section>
    {:else if active === 'dataset'}
      <section class="stage"><div><p class="eyebrow">强制审核门槛</p><h2>生成训练标签</h2><p>所有候选必须处于 accepted 或 rejected，才会生成并验证 PaddleOCR recognition 标签。</p></div><button class="button-primary" disabled={!batch || busy} on:click={finalize}>验证并生成 labels</button></section>
    {:else}
      <section class="stage training"><div><p class="eyebrow">LOCAL CPU / NO PUBLISH</p><h2>运行 Smoke 训练</h2><p>训练在独立子进程执行，不会上传模型或修改正式训练数据集。</p></div><div class="flex flex-wrap gap-3"><button class="button-secondary" disabled={!batch} on:click={refreshTraining}>刷新状态</button><button class="button-primary" disabled={!batch || busy} on:click={startTraining}>启动 CPU Smoke</button></div>{#if training}<pre>{JSON.stringify(training, null, 2)}</pre>{/if}</section>
    {/if}
  </section>
</main>
