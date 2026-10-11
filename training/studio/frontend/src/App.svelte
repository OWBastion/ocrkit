<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte'

  type ReviewCounts = { total: number; accepted: number; pending: number; rejected: number; teacher_eligible?: number }
  type Batch = { batch_id: string; sources: number; train_sources: number; holdout_sources: number; quality_warnings: number; layout_version: string; active_dataset_revision?: string | null; screenshot_set?: { set_id: string; version: number; finalized_at?: string; imported_at?: string } | null; review?: ReviewCounts }
  type JevSuggestion = { action: 'accept' | 'reject' | 'manual' | 'error'; selected_option?: string | null; confidence?: number | null; status?: string; reason_code?: string }
  type JevState = { status?: string; log_tail?: string; processed?: number; saved?: number; errors?: number; jev_adopted?: { accepted: number; rejected: number } | null }
  type PipelineState = { status?: string; stage?: string; completed?: number; total?: number; new_screenshots?: number; batch_id?: string | null; jev_errors?: number; agreement_adopted?: number; jev_adopted?: { accepted: number; rejected: number } | null; review?: ReviewCounts; log_tail?: string }
  type PipelinePreview = { new: number; recent: number; latest_batch: string | null }
  type Row = { category?: 'event' | 'achievement' | 'other'; jev_suggestion?: JevSuggestion; crop: string; roi: string; layout_version?: string; accuracy_feedback?: 'accurate' | 'inaccurate' | null; review_status: string; review_method?: 'automatic' | 'human' | 'pending'; auto_accept_reason?: string | null; auto_reject_reason?: string | null; candidate_text?: string; transcription?: string; suggested_transcription?: string | null; confidence?: number; candidate_confidence?: number; rapidocr_text?: string; rapidocr_confidence?: number; vision_text?: string; vision_confidence?: number; teacher_model_version?: string | null; teacher_text?: string | null; teacher_confidence?: number | null; teacher_suggestion?: boolean; teacher_auto_accept_eligible?: boolean }
  type CropZoom = 'auto' | 1 | 2 | 3 | 4
  type TrainingState = {
    status?: string
    backend?: string
    output_run_dir?: string
    kaggle_kernel?: string
    checkpoint?: string
    field_accuracy?: number | null
    verdict?: string | null
    reasons?: string[] | null
    override?: { reason: string } | null
    base?: { kind: 'official' | 'checkpoint'; name?: string }
    data?: { batches: string[]; train_samples?: number; holdout_samples?: number }
    pid?: number
    log?: string
    log_tail?: string
    command?: string[]
  }
  type ReleaseHistoryEntry = { manifest_key?: string; action?: string; verified_at?: string | null }
  type ReleaseComparison = {
    eligible: boolean
    reasons: string[]
    candidate: { version?: string; manifest_key?: string; channel_key?: string }
    stable: { version?: string; manifest_key?: string; channel_key?: string; history?: ReleaseHistoryEntry[] }
    comparison?: { field_accuracy?: { candidate?: number | null; stable?: number | null; delta?: number | null }; run_code_accuracy?: { candidate?: number | null; stable?: number | null; delta?: number | null }; holdout_accuracy?: { candidate?: number | null; stable?: number | null; delta?: number | null } }
  }
  type ResumeCheckpoint = { path: string; name: string }
  type TrainingOptions = { base_weights: { path: string; name: string }[]; data_batches: { batch_id: string; train: number; holdout: number }[] }
  type RemoteConfig = { configured: boolean; bucket: string; allowed_prefixes: string[]; max_objects?: number; max_object_bytes?: number }
  type RemoteObject = { key: string; size: number; etag?: string | null; last_modified?: string | null }
  type RemoteSortField = 'date' | 'size' | 'name'
  type RemoteSortOrder = 'desc' | 'asc'
  type RemoteViewMode = 'list' | 'grid'
  type DownloadProgress = {
    stage: 'downloading' | 'creating' | 'completed'
    completed: number
    total: number
    currentKey?: string
    message?: string
  }
  type RoiMeta = { name: string; dotColor: string }

  // Regions Studio collects. The central banner is intentionally absent: it overlaps the side panels.
  const ROI_INFO: Record<string, RoiMeta> = {
    left_panel: { name: '左侧统计', dotColor: '#0284c7' },
    achievement_panel: { name: '事件与成就', dotColor: '#b45309' },
    run_code_panel: { name: '运行代码', dotColor: '#6d28d9' },
    run_code_right_panel: { name: '右侧代码', dotColor: '#7c3aed' },
    right_panel: { name: '右上地图', dotColor: '#b91c1c' },
    server_load_panel: { name: '服务器负载', dotColor: '#be123c' },
    bottom_left_hero: { name: '玩家名', dotColor: '#4338ca' },
    ai_watermark: { name: 'AI 标记', dotColor: '#0f766e' },
  }

  function getRoiInfo(code: string): RoiMeta {
    return ROI_INFO[code] || { name: code, dotColor: '#52525b' }
  }

  let batches: Batch[] = []
  let batch: Batch | null = null
  let selected: Row | null = null
  let rows: Row[] = []
  let split = 'train'
  let status = 'pending'
  let roiFilter = 'all'
  let files: File[] = []
  let holdoutRatio = 0.2
  let notice = ''
  let error = ''
  let toastVisible = false
  let busy = false
  let active = 'import'
  let training: TrainingState | null = null
  let publication: TrainingState | null = null
  let releaseComparison: ReleaseComparison | null = null
  let releaseConfirmed = false
  let rollbackManifest = ''
  let releaseBusy = false
  let publishConfirmed = false
  let resumeCheckpoints: ResumeCheckpoint[] = []
  let trainingOptions: TrainingOptions | null = null
  let baseWeights = ''
  let includedBatches = new Set<string>()
  let resumeCheckpoint = ''
  let trainingEpochs = 10
  let trainingUpdatedAt = ''
  let trainingPolling = false
  let followLog = true
  let followPublishLog = true
  let logEl: HTMLPreElement | null = null
  let publishLogEl: HTMLPreElement | null = null
  let remoteConfig: RemoteConfig | null = null
  let remotePrefix = ''
  let remoteObjects: RemoteObject[] = []
  let remoteCursor: string | null = null
  let remoteSelected = new Set<string>()
  let remoteLoading = false
  let remoteFilterText = ''
  let remoteSortField: RemoteSortField = 'date'
  let remoteSortOrder: RemoteSortOrder = 'desc'
  let remoteViewMode: RemoteViewMode = 'grid'
  let remoteDays = '30'
  let remoteFeedback: Record<string, 'accurate' | 'inaccurate'> = {}
  let feedbackAvailable = false
  let feedbackLoading = false
  let feedbackFilter = 'all'
  let remoteImportMode = 'new'
  let generateAfterImport = true
  let trainingBackend = 'kaggle'
  let jevConfigured = false
  let jevState: JevState | null = null
  let jevPollTimer: ReturnType<typeof setInterval> | null = null
  let build = ''
  let pipeline: PipelineState | null = null
  let pipelinePreview: PipelinePreview | null = null
  let pipelineError = ''
  let pipelinePollTimer: ReturnType<typeof setInterval> | null = null
  let autoPrepare = false
  const AUTO_PREPARE_KEY = 'ocrkit-studio-auto-prepare'
  const pipelineStages = [['listing', '查找新截图'], ['downloading', '下载'], ['candidates', '切片 + OCR'], ['jev', 'JEV 建议'], ['done', '完成']] as const
  let hoverPreviewObject: RemoteObject | null = null
  let hoverPreviewTimeout: ReturnType<typeof setTimeout> | null = null
  let previewModalObject: RemoteObject | null = null
  let showRecreateModal = false
  let downloadProgress: DownloadProgress | null = null
  let cropZoom: CropZoom = 'auto'
  let cropNatural = { w: 0, h: 0 }
  let trainingPollTimer: ReturnType<typeof setInterval> | null = null
  let toastTimer: ReturnType<typeof setTimeout> | null = null
  let lastFinalize: { train: number; holdout: number } | null = null
  let lastExportPath = ''
  let setVersionInput: number | undefined = undefined
  let setImportBusy = false
  const TRAINING_POLL_MS = 2000
  const TOAST_MS = 4200
  const supportedClipboardTypes = new Set(['image/png', 'image/jpeg', 'image/webp'])

  function orderForReview(list: Row[]): Row[] {
    return [...list].sort((a, b) => Number(b.accuracy_feedback === 'inaccurate') - Number(a.accuracy_feedback === 'inaccurate'))
  }

  $: displayedRows = orderForReview(roiFilter === 'all' ? rows : rows.filter((row) => row.roi === roiFilter))

  const nav = [
    ['import', '1', '准备'],
    ['candidates', '2', '候选'],
    ['review', '3', '复核'],
    ['dataset', '4', '标签'],
    ['training', '5', '训练'],
  ] as const

  function openStep(step: (typeof nav)[number][0]) {
    if (step !== 'training') stopTrainingPoll()
    active = step
    if (step === 'review' && batch) { void refreshReview(); void resumeJevPoll() }
    if (step === 'training' && batch) void loadTrainingStep()
  }

  async function selectBatch(batchId: string) {
    batch = batches.find((item) => item.batch_id === batchId) || null
    stopJevPoll()
    jevState = null
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
    if (!batch) return '一键准备新截图 → 复核不确定项 → Kaggle 训练'
    const review = batch.review
    if (!review || review.total === 0) return '下一步：生成候选'
    if (review.pending > 0) return `下一步：复核剩余 ${review.pending} 条`
    return '下一步：选择训练方式并启动训练'
  }

  async function request<T>(url: string, init?: RequestInit): Promise<T> {
    const response = await fetch(url, init)
    if (!response.ok) {
      const body = await response.json().catch(() => ({})) as { detail?: string }
      throw new Error(body.detail || `请求失败 (${response.status})`)
    }
    return response.json() as Promise<T>
  }

  function clearToastTimer() {
    if (toastTimer) {
      clearTimeout(toastTimer)
      toastTimer = null
    }
  }

  function dismissToast() {
    clearToastTimer()
    toastVisible = false
    notice = ''
    error = ''
  }

  function message(value: string, isError = false) {
    clearToastTimer()
    error = isError ? value : ''
    notice = isError ? '' : value
    toastVisible = true
    // Success banners auto-dismiss; errors stay until dismissed.
    if (!isError) {
      toastTimer = setTimeout(() => {
        toastVisible = false
        notice = ''
      }, TOAST_MS)
    }
  }

  function toastKind() {
    return error ? 'error' : 'success'
  }

  function toastTitle() {
    return error ? '操作未完成' : '已完成'
  }

  function toastBody() {
    return error || notice
  }

  async function refreshBatches(selectId?: string) {
    batches = await request<Batch[]>('/api/batches')
    batch = batches.find((item) => item.batch_id === (selectId || batch?.batch_id)) || null
  }

  function formatBytes(value: number) {
    if (value < 1024 * 1024) return `${Math.round(value / 1024)} KiB`
    return `${(value / 1024 / 1024).toFixed(1)} MiB`
  }

  function formatDate(isoString?: string | null): string {
    if (!isoString) return '—'
    const date = new Date(isoString)
    if (Number.isNaN(date.getTime())) return isoString
    const pad = (n: number) => String(n).padStart(2, '0')
    const year = date.getFullYear()
    const month = pad(date.getMonth() + 1)
    const day = pad(date.getDate())
    const hours = pad(date.getHours())
    const minutes = pad(date.getMinutes())
    const seconds = pad(date.getSeconds())
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`
  }

  function splitKey(key: string): { dir: string; name: string; ext: string } {
    const lastSlash = key.lastIndexOf('/')
    const dir = lastSlash >= 0 ? key.slice(0, lastSlash + 1) : ''
    const name = lastSlash >= 0 ? key.slice(lastSlash + 1) : key
    const lastDot = name.lastIndexOf('.')
    const ext = lastDot >= 0 ? name.slice(lastDot + 1).toUpperCase() : ''
    return { dir, name, ext }
  }

  $: filteredRemoteObjects = (() => {
    let list = [...remoteObjects]
    if (feedbackFilter !== 'all') list = list.filter((item) => feedbackFilter === 'unknown' ? !remoteFeedback[item.key] : remoteFeedback[item.key] === feedbackFilter)
    const query = remoteFilterText.trim().toLowerCase()
    if (query) {
      list = list.filter((item) => {
        const key = item.key.toLowerCase()
        const dateStr = formatDate(item.last_modified).toLowerCase()
        const sizeStr = formatBytes(item.size).toLowerCase()
        return key.includes(query) || dateStr.includes(query) || sizeStr.includes(query)
      })
    }
    list.sort((a, b) => {
      let result = 0
      if (remoteSortField === 'date') {
        const timeA = a.last_modified ? new Date(a.last_modified).getTime() : 0
        const timeB = b.last_modified ? new Date(b.last_modified).getTime() : 0
        result = timeA - timeB
      } else if (remoteSortField === 'size') {
        result = a.size - b.size
      } else if (remoteSortField === 'name') {
        result = a.key.localeCompare(b.key)
      }
      return remoteSortOrder === 'desc' ? -result : result
    })
    return list
  })()

  $: isAllSelected = filteredRemoteObjects.length > 0 && filteredRemoteObjects.every((item) => remoteSelected.has(item.key))
  $: isPartiallySelected = !isAllSelected && filteredRemoteObjects.some((item) => remoteSelected.has(item.key))
  $: selectedTotalBytes = [...remoteSelected].reduce((sum, key) => {
    const obj = remoteObjects.find((o) => o.key === key)
    return sum + (obj?.size || 0)
  }, 0)

  async function loadRemoteStatus() {
    remoteConfig = await request<RemoteConfig>('/api/r2/status')
    if (remoteConfig.configured && !remotePrefix) remotePrefix = remoteConfig.allowed_prefixes[0] || ''
  }

  function feedbackLabel(key: string) {
    return remoteFeedback[key] === 'inaccurate' ? '反馈：识别不准确' : remoteFeedback[key] === 'accurate' ? '反馈：识别准确' : '无准确反馈'
  }

  async function loadFeedback(since: string | null) {
    feedbackLoading = true
    try {
      const query = new URLSearchParams()
      if (since) query.set('since', since)
      const result = await request<{ available: boolean; marks: Record<string, 'accurate' | 'inaccurate'> }>(`/api/r2/feedback?${query}`)
      remoteFeedback = result.marks
      feedbackAvailable = result.available
    } catch { feedbackAvailable = false; remoteFeedback = {} } finally { feedbackLoading = false }
  }

  async function loadRemoteImages() {
    if (!remoteConfig?.configured) return message('Studio 尚未配置 R2 远程数据源。', true)
    if (!remotePrefix) return message('请输入或选择一个 R2 prefix。', true)
    remoteLoading = true
    remoteSelected = new Set()
    remoteObjects = []
    try {
      let cursor: string | null = null
      const since = remoteDays === 'all' ? null : new Date(Date.now() - Number(remoteDays) * 86400000).toISOString()
      void loadFeedback(since)
      do {
        const query = new URLSearchParams({ prefix: remotePrefix })
        if (since) query.set('since', since)
        if (cursor) query.set('cursor', cursor)
        const result = await request<{ objects: RemoteObject[]; next_cursor: string | null }>(`/api/r2/images?${query}`)
        remoteObjects = [...remoteObjects, ...result.objects]
        cursor = result.next_cursor
        remoteCursor = cursor
      } while (cursor)
      message(`已加载 ${remoteObjects.length} 张截图，按时间从新到旧排列。`)
    } catch (cause) { message(cause instanceof Error ? cause.message : '读取 R2 图片失败', true) } finally { remoteLoading = false }
  }

  function toggleRemoteSelection(key: string) {
    const next = new Set(remoteSelected)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    remoteSelected = next
  }

  function toggleSelectAll() {
    const next = new Set(remoteSelected)
    if (isAllSelected) {
      for (const item of filteredRemoteObjects) {
        next.delete(item.key)
      }
    } else {
      for (const item of filteredRemoteObjects) {
        next.add(item.key)
      }
    }
    remoteSelected = next
  }

  function invertSelection() {
    const next = new Set(remoteSelected)
    for (const item of filteredRemoteObjects) {
      if (next.has(item.key)) {
        next.delete(item.key)
      } else {
        next.add(item.key)
      }
    }
    remoteSelected = next
  }

  function clearSelection() {
    remoteSelected = new Set()
  }

  function showHoverPreview(object: RemoteObject) {
    if (hoverPreviewTimeout) clearTimeout(hoverPreviewTimeout)
    hoverPreviewTimeout = setTimeout(() => {
      hoverPreviewObject = object
    }, 150)
  }

  function hideHoverPreview() {
    if (hoverPreviewTimeout) clearTimeout(hoverPreviewTimeout)
    hoverPreviewTimeout = setTimeout(() => {
      hoverPreviewObject = null
    }, 80)
  }

  function openPreviewModal(object: RemoteObject) {
    hoverPreviewObject = null
    previewModalObject = object
  }

  function closePreviewModal() {
    previewModalObject = null
  }

  function prevPreview() {
    if (!previewModalObject) return
    const index = filteredRemoteObjects.findIndex((item) => item.key === previewModalObject?.key)
    if (index > 0) {
      previewModalObject = filteredRemoteObjects[index - 1]
    } else if (filteredRemoteObjects.length > 0) {
      previewModalObject = filteredRemoteObjects[filteredRemoteObjects.length - 1]
    }
  }

  function nextPreview() {
    if (!previewModalObject) return
    const index = filteredRemoteObjects.findIndex((item) => item.key === previewModalObject?.key)
    if (index >= 0 && index < filteredRemoteObjects.length - 1) {
      previewModalObject = filteredRemoteObjects[index + 1]
    } else if (filteredRemoteObjects.length > 0) {
      previewModalObject = filteredRemoteObjects[0]
    }
  }

  function onKeyDown(event: KeyboardEvent) {
    if (showRecreateModal) {
      if (event.key === 'Escape' && !busy) {
        event.preventDefault()
        closeRecreateModal()
      } else if (event.key === 'Enter' && !busy) {
        event.preventDefault()
        void confirmRecreateCandidates()
      }
      return
    }

    if (previewModalObject) {
      if (event.key === 'Escape') {
        closePreviewModal()
      } else if (event.key === 'ArrowLeft') {
        prevPreview()
      } else if (event.key === 'ArrowRight') {
        nextPreview()
      } else if (event.key === ' ' || event.key === 'Enter') {
        if ((event.target as HTMLElement)?.tagName !== 'BUTTON' && (event.target as HTMLElement)?.tagName !== 'INPUT') {
          event.preventDefault()
          toggleRemoteSelection(previewModalObject.key)
        }
      }
      return
    }

    if (active === 'review' && selected && batch && !busy) {
      const isCmdOrCtrl = event.metaKey || event.ctrlKey
      if (isCmdOrCtrl && event.key === 'Enter') {
        event.preventDefault()
        void save('accepted')
      } else if (event.altKey && (event.key === 'Backspace' || event.key === 'Delete')) {
        event.preventDefault()
        void save('rejected')
      } else if (event.altKey && event.key === 'ArrowUp') {
        event.preventDefault()
        selectPrevCandidate()
      } else if (event.altKey && event.key === 'ArrowDown') {
        event.preventDefault()
        selectNextCandidate()
      }
    }
  }

  async function streamImport(endpoint: string, payload: object, onChunk: (data: any) => void) {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({})) as { detail?: string }
      throw new Error(errorBody.detail || `请求失败 (${response.status})`)
    }
    const reader = response.body?.getReader()
    if (!reader) throw new Error('无法读取响应流')
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (line.trim()) {
          const parsed = JSON.parse(line)
          onChunk(parsed)
        }
      }
    }
    if (buffer.trim()) {
      const parsed = JSON.parse(buffer)
      onChunk(parsed)
    }
  }

  async function importRemoteImages() {
    if (!remoteSelected.size) return message('先选择至少一张 R2 截图。', true)
    const addingToExistingBatch = remoteImportMode === 'append' && Boolean(batch)
    const totalKeys = remoteSelected.size
    busy = true
    downloadProgress = {
      stage: 'downloading',
      completed: 0,
      total: totalKeys,
      message: `准备并行下载 ${totalKeys} 张截图...`,
    }

    try {
      const endpoint = addingToExistingBatch && batch ? `/api/batches/${batch.batch_id}/remote-sources/stream` : '/api/batches/r2/stream'
      let finalResult: any = null

      await streamImport(endpoint, { keys: [...remoteSelected], holdout_ratio: holdoutRatio, accuracy_marks: Object.fromEntries([...remoteSelected].filter((key) => remoteFeedback[key]).map((key) => [key, remoteFeedback[key]])) }, (chunk) => {
        if (chunk.type === 'progress') {
          downloadProgress = {
            stage: chunk.stage,
            completed: chunk.completed,
            total: chunk.total,
            currentKey: chunk.current_key,
            message: chunk.message,
          }
        } else if (chunk.type === 'error') {
          throw new Error(chunk.detail || '导入失败')
        } else if (chunk.type === 'done') {
          finalResult = chunk
        }
      })

      if (!finalResult) throw new Error('未收到批次处理结果')

      const batchId = finalResult.batch?.batch_id || batch?.batch_id
      if (batchId) await refreshBatches(batchId)
      remoteSelected = new Set()
      message(addingToExistingBatch ? `已从 R2 加入 ${finalResult.added || totalKeys} 张截图。` : `已用 R2 截图创建批次（${finalResult.batch?.sources || totalKeys} 张）。`)
      active = 'candidates'
      if (generateAfterImport) await candidates()
    } catch (cause) {
      message(cause instanceof Error ? cause.message : '导入 R2 图片失败', true)
    } finally {
      busy = false
      downloadProgress = null
    }
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

    const first = orderForReview(roiFilter === 'all' ? rows : rows.filter((row) => row.roi === roiFilter))[0]
    if (first) selectCandidate(first)
    else selected = null
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

  async function addFilesToBatch() {
    if (!batch) return message('先选择要补充截图的批次。', true)
    if (!files.length) return message('先选择至少一张截图。', true)
    busy = true
    try {
      const data = new FormData()
      files.forEach((file) => data.append('files', file))
      const result = await request<{ added: number; batch: Batch }>(`/api/batches/${batch.batch_id}/sources`, { method: 'POST', body: data })
      await refreshBatches(result.batch.batch_id)
      files = []
      message(`已加入 ${result.added} 张新截图；请生成候选并复核新增切片。`)
      active = 'candidates'
    } catch (cause) { message(cause instanceof Error ? cause.message : '追加截图失败', true) } finally { busy = false }
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
      const result = await request<{ review: ReviewCounts; summary: { reused_existing_candidates?: boolean; negative_auto_rejected?: number; deduplicated?: number; teacher_model_version?: string | null; teacher_suggestions?: number; teacher_auto_accept_eligible?: number } }>(`/api/batches/${batch.batch_id}/candidates`, { method: 'POST' })
      batch = { ...batch, review: result.review }
      const teacherHint = result.summary.teacher_model_version
        ? `上一版模型 ${result.summary.teacher_model_version} 已提供 ${result.summary.teacher_suggestions || 0} 条建议；Train 会自动接受与 RapidOCR 高置信一致的结果。`
        : ''
      const negativeHint = result.summary.negative_auto_rejected
        ? `已根据历史人工拒绝排除 ${result.summary.negative_auto_rejected} 条重复错误候选。`
        : ''
      const duplicateHint = result.summary.deduplicated
        ? `已合并 ${result.summary.deduplicated} 条重叠 ROI 重复候选。`
        : ''
      message(result.summary.reused_existing_candidates ? `已打开现有候选，进入复核。${duplicateHint}${negativeHint}` : `候选已生成。${duplicateHint}${negativeHint}${teacherHint}`)
      active = 'review'
      await resumeJevPoll()
      await refreshReview()
      if (displayedRows[0]) selectCandidate(displayedRows[0])
    } catch (cause) { message(cause instanceof Error ? cause.message : '候选生成失败', true) } finally { busy = false }
  }

  function openRecreateModal() {
    if (!batch) return message('先选择或创建批次。', true)
    showRecreateModal = true
  }

  function closeRecreateModal() {
    if (busy) return
    showRecreateModal = false
  }

  async function confirmRecreateCandidates() {
    if (!batch || busy) return
    busy = true
    try {
      const result = await request<{
        review: ReviewCounts
        summary: {
          revision_id?: string
          inherited_manual_accepted?: number
          inherited_manual_rejected?: number
          unmatched_manual_decisions?: number
          train_candidates?: number
          holdout_candidates?: number
        }
      }>(`/api/batches/${batch.batch_id}/candidates/recreate`, { method: 'POST' })
      batch = { ...batch, review: result.review }
      const accepted = result.summary.inherited_manual_accepted || 0
      const rejected = result.summary.inherited_manual_rejected || 0
      const unmatched = result.summary.unmatched_manual_decisions || 0
      message(`候选已重建（${result.summary.revision_id || '新版本'}）。已继承人工接受 ${accepted} 条、拒绝 ${rejected} 条；${unmatched} 条无法安全匹配，已重新待复核。`)
      showRecreateModal = false
      active = 'review'
      await resumeJevPoll()
      await refreshReview()
      if (displayedRows[0]) selectCandidate(displayedRows[0])
    } catch (cause) {
      message(cause instanceof Error ? cause.message : '候选重建失败', true)
    } finally {
      busy = false
    }
  }

  async function acceptTeacherSuggestions() {
    if (!batch || split !== 'train' || !(batch.review?.teacher_eligible ?? 0)) return
    busy = true
    try {
      const result = await request<{ result: { accepted: number }; counts: ReviewCounts }>(`/api/batches/${batch.batch_id}/review/accept-teacher`, { method: 'POST' })
      batch = { ...batch, review: result.counts }
      await refreshReview()
      if (displayedRows[0]) selectCandidate(displayedRows[0])
      message(`已批量接受 ${result.result.accepted} 条上一版模型 + RapidOCR 一致项；holdout 未自动接受。`)
    } catch (cause) {
      message(cause instanceof Error ? cause.message : '批量接受上一版模型建议失败', true)
    } finally {
      busy = false
    }
  }

  type Engine = { name: string; text: string; confidence?: number | null }
  type Suggestion = { text: string; label: string; hint: string; tone: 'good' | 'warn' | 'bad' | 'muted'; reject: boolean }

  const norm = (value?: string | null) => (value || '').normalize('NFKC').replace(/\s+/g, ' ').trim()

  function enginesOf(row: Row): Engine[] {
    const all: Engine[] = [
      { name: 'RapidOCR', text: row.rapidocr_text || '', confidence: row.rapidocr_confidence },
      { name: 'Vision', text: row.vision_text || '', confidence: row.vision_confidence },
      { name: '上一版', text: row.teacher_text || '', confidence: row.teacher_confidence },
    ]
    return all.filter((engine) => norm(engine.text))
  }

  // A name with a duration, e.g. "心之钢 (27 秒)", is the event; anything else in the stack starts as an achievement.
  const defaultCategory = (text: string): 'event' | 'achievement' => (/[（(]\s*\d+\s*秒\s*[）)]/.test(text) ? 'event' : 'achievement')
  const CATEGORIES = [['event', '事件'], ['achievement', '成就'], ['other', '其它']] as const

  // One answer to start from. JEV settles disagreements between engines; it never overrides engines that already agree.
  function suggestionFor(row: Row): Suggestion {
    const engines = enginesOf(row)
    const jev = row.jev_suggestion?.status === 'ok' ? row.jev_suggestion : undefined
    const groups = new Map<string, Engine[]>()
    for (const engine of engines) groups.set(norm(engine.text), [...(groups.get(norm(engine.text)) || []), engine])
    const agreed = [...groups.values()].find((group) => group.length >= 2)
    const best = [...engines].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))[0]
    const jevText = jev?.action === 'accept' ? jev.selected_option || '' : ''
    const text = agreed?.[0].text || jevText || row.suggested_transcription || best?.text || row.candidate_text || ''
    const percent = confidenceLabel(jev?.confidence)
    if (jev?.action === 'reject') return { text, label: `JEV ${percent}`, hint: 'JEV 认为这不是有效切片，建议拒绝', tone: 'bad', reject: true }
    if (jev?.action === 'manual' && !agreed) return { text, label: `JEV ${percent}`, hint: 'JEV 认为候选都不对，请对照图片修改', tone: 'warn', reject: false }
    if (agreed) return { text, label: `${agreed.length} 个引擎一致`, hint: agreed.map((engine) => engine.name).join(' · '), tone: 'good', reject: false }
    if (jevText) return { text, label: `JEV ${percent}`, hint: '引擎有分歧，JEV 选了这一条', tone: 'good', reject: false }
    if (best) return { text, label: `${best.name} ${confidenceLabel(best.confidence)}`, hint: '只有一个引擎有结果，请核对', tone: 'warn', reject: false }
    return { text, label: '无建议', hint: '请对照图片手动输入', tone: 'muted', reject: false }
  }

  // Distinct other readings, each with who produced it.
  function alternativesFor(row: Row, current?: string | null): { text: string; who: string }[] {
    const found = new Map<string, { text: string; who: string[] }>()
    const add = (text: string | null | undefined, who: string) => {
      const key = norm(text)
      if (!key) return
      const entry = found.get(key) || { text: text as string, who: [] }
      if (!entry.who.includes(who)) entry.who.push(who)
      found.set(key, entry)
    }
    for (const engine of enginesOf(row)) add(engine.text, engine.name)
    if (row.jev_suggestion?.status === 'ok' && row.jev_suggestion.action === 'accept') add(row.jev_suggestion.selected_option, 'JEV')
    add(row.candidate_text, '候选')
    return [...found.entries()].filter(([key]) => key !== norm(current)).map(([, entry]) => ({ text: entry.text, who: entry.who.join(' · ') }))
  }

  let transcriptionEl: HTMLInputElement | null = null
  let focusedCrop = ''
  $: if (selected && transcriptionEl && selected.crop !== focusedCrop && typeof matchMedia !== 'undefined' && matchMedia('(pointer: fine)').matches) {
    focusedCrop = selected.crop
    transcriptionEl.focus()
    transcriptionEl.select()
  }

  function onTranscriptionKey(event: KeyboardEvent) {
    if (event.key !== 'Enter' || event.isComposing || event.shiftKey || event.metaKey || event.ctrlKey || event.altKey) return
    event.preventDefault()
    save('accepted')
  }

  function toggleCropZoom() {
    cropZoom = cropZoom === 'auto' ? 2 : cropZoom === 2 ? 4 : 'auto'
  }

  async function trainCurrentBatch() {
    if (!batch) return
    trainingBackend = 'kaggle'
    active = 'training'
    await loadTrainingStep()
  }

  function confidenceLabel(value?: number | null) {
    if (value == null || Number.isNaN(value)) return '—'
    return `${Math.round(value * 100)}%`
  }

  function selectCandidate(row: Row, event?: MouseEvent) {
    const text = row.transcription || suggestionFor(row).text
    selected = { ...row, transcription: text, category: row.category ?? (row.roi === 'achievement_panel' ? defaultCategory(text) : undefined) }
    cropZoom = 'auto'
    cropNatural = { w: 0, h: 0 }
    const target = event?.currentTarget
    if (target instanceof HTMLElement) target.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }

  function selectPrevCandidate() {
    if (!selected || !displayedRows.length) return
    const currentIndex = displayedRows.findIndex((row) => row.crop === selected?.crop)
    if (currentIndex > 0) {
      selectCandidate(displayedRows[currentIndex - 1])
    } else {
      selectCandidate(displayedRows[displayedRows.length - 1])
    }
  }

  function selectNextCandidate() {
    if (!selected || !displayedRows.length) return
    const currentIndex = displayedRows.findIndex((row) => row.crop === selected?.crop)
    if (currentIndex >= 0 && currentIndex < displayedRows.length - 1) {
      selectCandidate(displayedRows[currentIndex + 1])
    } else {
      selectCandidate(displayedRows[0])
    }
  }

  function onRoiFilterChange() {
    if (selected && roiFilter !== 'all' && selected.roi !== roiFilter) {
      const match = displayedRows[0]
      if (match) selectCandidate(match)
      else selected = null
    } else if (!selected && displayedRows.length > 0) {
      selectCandidate(displayedRows[0])
    }
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

  function applyEngineText(text?: string | null) {
    if (!selected || !text) return
    selected = { ...selected, transcription: text }
  }

  let saveChain: Promise<unknown> = Promise.resolve()

  // Advance immediately; the write follows in order and the list is reloaded only if it fails.
  function save(statusValue: 'accepted' | 'rejected') {
    if (!batch || !selected) return
    const row = selected
    const batchId = batch.batch_id
    const transcription = (row.transcription || row.candidate_text || '').trim()
    if (statusValue === 'accepted' && !transcription) return message('转写不能为空。', true)
    const index = displayedRows.findIndex((item) => item.crop === row.crop)
    const next = displayedRows[index + 1] ?? displayedRows[index - 1] ?? null
    const wasPending = row.review_status === 'pending'
    rows = status === 'pending'
      ? rows.filter((item) => item.crop !== row.crop)
      : rows.map((item) => item.crop === row.crop ? { ...item, review_status: statusValue, transcription: statusValue === 'accepted' ? transcription : undefined } : item)
    if (batch.review && wasPending) {
      batch = { ...batch, review: { ...batch.review, pending: Math.max(0, batch.review.pending - 1), [statusValue]: batch.review[statusValue] + 1 } }
    }
    const following = next && rows.find((item) => item.crop === next.crop)
    if (following) selectCandidate(following)
    else selected = null
    const currentSplit = split
    saveChain = saveChain
      .then(() => request<{ row: Row; counts: ReviewCounts }>(`/api/batches/${batchId}/review`, {
        method: 'PUT',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ split: currentSplit, crop: row.crop, status: statusValue, transcription, category: row.roi === 'achievement_panel' ? row.category : undefined }),
      }))
      .then((result) => { if (batch?.batch_id === batchId) batch = { ...batch, review: result.counts } })
      .catch((cause) => {
        message(cause instanceof Error ? cause.message : '保存失败，已重新载入。', true)
        void refreshReview()
      })
  }

  async function refreshVision() {
    if (!batch) return message('先选择或创建批次。', true)
    busy = true
    try {
      const result = await request<{ review: ReviewCounts; summary: { rows: number; vision_covered: number; auto_accepted: number; auto_rejected: number; preserved_accepted: number; preserved_rejected: number } }>(`/api/batches/${batch.batch_id}/candidates/refresh-vision`, { method: 'POST' })
      batch = { ...batch, review: result.review }
      message(`已补回 Vision：${result.summary.vision_covered}/${result.summary.rows} 条有结果，自动排除 ${result.summary.auto_rejected} 条位置不匹配项；保留 ${result.summary.preserved_accepted} 条人工接受和 ${result.summary.preserved_rejected} 条人工拒绝。`)
      active = 'review'
      await resumeJevPoll()
      await refreshReview()
      if (displayedRows[0]) selectCandidate(displayedRows[0])
    } catch (cause) { message(cause instanceof Error ? cause.message : '补回 Vision 结果失败', true) } finally { busy = false }
  }

  async function refreshTeacher() {
    if (!batch) return message('先选择或创建批次。', true)
    busy = true
    try {
      const result = await request<{ review: ReviewCounts; summary: { rows: number; teacher_model_version?: string | null; teacher_covered: number; teacher_suggestions: number; teacher_auto_accept_eligible: number; teacher_auto_accepted: number; auto_rejected: number; preserved_accepted: number; preserved_rejected: number } }>(`/api/batches/${batch.batch_id}/candidates/refresh-teacher`, { method: 'POST' })
      batch = { ...batch, review: result.review }
      message(`已补回上一版模型 ${result.summary.teacher_model_version || '未知版本'}：覆盖 ${result.summary.teacher_covered}/${result.summary.rows} 条，自动接受 ${result.summary.teacher_auto_accepted} 条，自动排除 ${result.summary.auto_rejected} 条位置不匹配项；保留 ${result.summary.preserved_accepted} 条人工接受和 ${result.summary.preserved_rejected} 条人工拒绝。`)
      active = 'review'
      await resumeJevPoll()
      await refreshReview()
      if (displayedRows[0]) selectCandidate(displayedRows[0])
    } catch (cause) { message(cause instanceof Error ? cause.message : '补回上一版模型结果失败', true) } finally { busy = false }
  }

  async function finalize() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try {
      const result = await request<Record<string, number>>(`/api/batches/${batch.batch_id}/finalize`, { method: 'POST' })
      lastFinalize = { train: result.validated_train, holdout: result.validated_holdout }
      message(`标签已生成：train ${result.validated_train}，holdout ${result.validated_holdout}。`)
    } catch (cause) { message(cause instanceof Error ? cause.message : '生成标签失败', true) } finally { busy = false }
  }

  async function exportDataset() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try {
      const result = await request<{ export_dir: string; validated_train: number; validated_holdout: number }>(`/api/batches/${batch.batch_id}/dataset/export`, { method: 'POST' })
      lastFinalize = { train: result.validated_train, holdout: result.validated_holdout }
      lastExportPath = result.export_dir
      message(`已导出私有数据集：${result.validated_train} train / ${result.validated_holdout} holdout。`)
    } catch (cause) { message(cause instanceof Error ? cause.message : '导出数据集失败', true) } finally { busy = false }
  }

  function datasetReadyHint() {
    if (!batch) return '先选择批次'
    const pending = batch.review?.pending ?? null
    if (pending == null) return '尚未生成候选'
    if (pending > 0) return `还有 ${pending} 条待复核`
    if ((batch.review?.accepted ?? 0) === 0 && (batch.review?.total ?? 0) > 0) return '至少需要一条已接受转写'
    return '可以生成 labels'
  }

  function canPublish() {
    return Boolean(batch && training?.status === 'completed' && publishConfirmed && publication?.status !== 'publishing' && !busy)
  }

  function publicationIsActive() {
    return publication?.status === 'publishing'
  }

  function trainingStatusLabel(value?: string) {
    switch (value) {
      case 'training': return '运行中'
      case 'publishing': return '发布中'
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

  function hasActiveBackgroundWork() {
    return trainingIsRunning(training?.status) || publication?.status === 'publishing'
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
      void refreshPublication({ silent: true })
    }, TRAINING_POLL_MS)
  }

  async function scrollLogToBottom() {
    await tick()
    if (followLog && logEl) logEl.scrollTop = logEl.scrollHeight
    if (followPublishLog && publishLogEl) publishLogEl.scrollTop = publishLogEl.scrollHeight
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
        if (!hasActiveBackgroundWork()) stopTrainingPoll()
        if (previousStatus === 'training' && trainingIsDone(training.status)) {
          message(training.status === 'completed' ? (training.backend === 'kaggle' ? '训练已成功结束，模型已下载到本地。' : '本地 Smoke 训练已成功结束。') : '训练已结束，请查看日志。')
        }
      }
      await scrollLogToBottom()
    } catch (cause) {
      stopTrainingPoll()
      if (!options?.silent) message(cause instanceof Error ? cause.message : '获取训练状态失败', true)
    }
  }

  // Which weights to start from and which finalized batches to learn from; every choice is shown on the run afterwards.
  async function refreshTrainingOptions() {
    if (!batch) return
    trainingOptions = await request<TrainingOptions>(`/api/batches/${batch.batch_id}/training/options`)
    if (baseWeights && !trainingOptions.base_weights.some((weights) => weights.path === baseWeights)) baseWeights = ''
    includedBatches = new Set([...includedBatches].filter((id) => trainingOptions?.data_batches.some((item) => item.batch_id === id)))
    if (!includedBatches.size && !trainingOptionsTouched) includedBatches = new Set(trainingOptions.data_batches.map((item) => item.batch_id))
  }

  let trainingOptionsTouched = false

  function toggleIncludedBatch(id: string) {
    trainingOptionsTouched = true
    const next = new Set(includedBatches)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    includedBatches = next
  }

  async function loadTrainingStep() {
    await Promise.all([refreshTraining(), refreshResumeCheckpoints(), refreshTrainingOptions(), refreshPublication(), refreshReleaseComparison({ silent: true })])
    if (trainingIsRunning(training?.status)) startTrainingPoll()
  }

  async function refreshPublication(options?: { silent?: boolean }) {
    if (!batch) return
    try {
      const previous = publication?.status
      publication = await request<TrainingState>(`/api/batches/${batch.batch_id}/publication`)
      if (publication.status === 'publishing' && active === 'training') startTrainingPoll()
      if (publication.status !== 'publishing' && !trainingIsRunning(training?.status)) stopTrainingPoll()
      // Surface completion even during silent polling so operators see the outcome.
      if (previous === 'publishing' && publication.status && publication.status !== 'publishing') {
        message(publication.status === 'completed' ? '模型已发布到 R2。' : '模型发布已结束，请查看发布日志。')
        void refreshReleaseComparison({ silent: true })
      }
      if (followPublishLog) await scrollLogToBottom()
    } catch (cause) {
      if (!options?.silent) message(cause instanceof Error ? cause.message : '获取发布状态失败', true)
    }
  }

  async function refreshReleaseComparison(options?: { silent?: boolean }) {
    try {
      releaseComparison = await request<ReleaseComparison>('/api/model-release/compare')
      const history = releaseComparison.stable.history || []
      if (rollbackManifest && !history.some((entry) => entry.manifest_key === rollbackManifest)) rollbackManifest = ''
    } catch (cause) {
      if (!options?.silent) message(cause instanceof Error ? cause.message : '获取模型比较结果失败', true)
    }
  }

  async function promoteStable() {
    if (!releaseConfirmed) return message('请确认候选证据后再晋级 stable。', true)
    releaseBusy = true
    try {
      await request('/api/model-release/promote', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ confirmed: true }) })
      releaseConfirmed = false
      message('候选已显式晋级 stable；生产容器需按现有部署流程重启。')
      await refreshReleaseComparison()
    } catch (cause) { message(cause instanceof Error ? cause.message : '模型晋级失败', true) } finally { releaseBusy = false }
  }

  async function rollbackStable() {
    if (!rollbackManifest) return message('请选择一个历史 verified manifest。', true)
    releaseBusy = true
    try {
      await request('/api/model-release/rollback', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ confirmed: true, manifest_key: rollbackManifest }) })
      message('stable 已回指历史 verified manifest；生产容器需按现有部署流程重启。')
      rollbackManifest = ''
      await refreshReleaseComparison()
    } catch (cause) { message(cause instanceof Error ? cause.message : '模型回滚失败', true) } finally { releaseBusy = false }
  }

  async function startPublication() {
    if (!batch || !publishConfirmed) return message('请确认已准备将新模型写入 R2。', true)
    busy = true
    try {
      await request<TrainingState>(`/api/batches/${batch.batch_id}/publication`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ confirmed: true }),
      })
      message('模型发布已启动：将重新评测、导出、上传并下载校验。')
      await refreshPublication()
      startTrainingPoll()
    } catch (cause) { message(cause instanceof Error ? cause.message : '启动 R2 发布失败', true) } finally { busy = false }
  }

  async function refreshResumeCheckpoints() {
    if (!batch) {
      resumeCheckpoints = []
      resumeCheckpoint = ''
      return
    }
    resumeCheckpoints = await request<ResumeCheckpoint[]>(`/api/batches/${batch.batch_id}/training/checkpoints`)
    if (resumeCheckpoint && !resumeCheckpoints.some((checkpoint) => checkpoint.path === resumeCheckpoint)) {
      resumeCheckpoint = ''
    }
  }

  let overrideReason = ''

  async function acceptOverride() {
    if (!batch) return
    busy = true
    try {
      await request<TrainingState>(`/api/batches/${batch.batch_id}/training/accept`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ reason: overrideReason.trim() }),
      })
      overrideReason = ''
      message('已按负责人决定放行；未通过的原因和理由已记入评测证据，现在可以发布。')
      await refreshTraining()
    } catch (cause) { message(cause instanceof Error ? cause.message : '放行失败', true) } finally { busy = false }
  }

  async function evaluateTrained() {
    if (!batch) return
    busy = true
    try {
      await request<TrainingState>(`/api/batches/${batch.batch_id}/training/evaluate`, { method: 'POST' })
      message('已开始评测已有检查点，不会重新训练。')
      followLog = true
      await refreshTraining()
      startTrainingPoll()
    } catch (cause) { message(cause instanceof Error ? cause.message : '评测启动失败', true) } finally { busy = false }
  }

  async function startTraining() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try {
      await request<TrainingState>(`/api/batches/${batch.batch_id}/training/${trainingBackend}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(trainingBackend === 'kaggle' ? { epochs: trainingEpochs, base_checkpoint: baseWeights || null, include_batches: [...includedBatches] } : { resume_checkpoint: resumeCheckpoint || null, epochs: trainingEpochs }),
      })
      message(`${trainingBackend === 'kaggle' ? 'Kaggle' : 'CPU Smoke'} 训练已启动，日志将自动刷新。`)
      followLog = true
      await refreshTraining()
      startTrainingPoll()
    } catch (cause) { message(cause instanceof Error ? cause.message : '训练启动失败', true) } finally { busy = false }
  }

  // --- 平台截图集导入（生产训练真值入口，#26） ---

  async function importScreenshotSet() {
    const version = Number(setVersionInput)
    if (!Number.isInteger(version) || version < 1) return message('输入截图集版本号（正整数）。', true)
    setImportBusy = true
    try {
      const result = await request<{ batch: Batch; report: { version: number; member_count: number } }>('/api/screenshot-sets/import', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ version, holdout_ratio: holdoutRatio }),
      })
      setVersionInput = undefined
      message(`截图集 v${result.report.version} 已导入为批次 ${result.batch.batch_id}（${result.report.member_count} 张源图）。`)
      await refreshBatches(result.batch.batch_id)
      active = 'candidates'
    } catch (cause) { message(cause instanceof Error ? cause.message : '截图集导入失败', true) } finally { setImportBusy = false }
  }

  function stopJevPoll() {
    if (jevPollTimer) clearInterval(jevPollTimer)
    jevPollTimer = null
  }

  async function refreshJev() {
    if (!batch) return
    jevState = await request<JevState>(`/api/batches/${batch.batch_id}/review/jev`)
    if (jevState.status !== 'reviewing') {
      stopJevPoll()
      await refreshReview({ keepSelection: true })
    }
  }

  async function resumeJevPoll() {
    await refreshJev()
    if (jevState?.status === 'reviewing' && !jevPollTimer) jevPollTimer = setInterval(() => { void refreshJev().catch(() => stopJevPoll()) }, 2000)
  }

  async function startJev() {
    if (!batch) return
    busy = true
    try {
      jevState = await request<JevState>(`/api/batches/${batch.batch_id}/review/jev`, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ image_tokens: 70 }),
      })
      stopJevPoll()
      jevPollTimer = setInterval(() => { void refreshJev().catch((cause) => { stopJevPoll(); message(String(cause), true) }) }, 2000)
      message('JEV 已开始生成建议，完成后人工确认标签。')
    } catch (cause) { message(cause instanceof Error ? cause.message : 'JEV 启动失败', true) } finally { busy = false }
  }

  // --- 一键准备：新 R2 截图 → 批次 → 切片/OCR → JEV → 采纳高置信 Train 标签 ---

  const pipelineRunning = () => pipeline?.status === 'preparing'

  function stopPipelinePoll() {
    if (pipelinePollTimer) clearInterval(pipelinePollTimer)
    pipelinePollTimer = null
  }

  async function refreshPipeline() {
    const wasRunning = pipelineRunning()
    pipeline = await request<PipelineState>('/api/pipeline/status')
    if (pipelineRunning()) {
      if (!pipelinePollTimer) pipelinePollTimer = setInterval(() => { void refreshPipeline().catch(stopPipelinePoll) }, 1500)
      return
    }
    stopPipelinePoll()
    if (wasRunning) {
      await refreshBatches(pipeline.batch_id || undefined)
      if (pipeline.status === 'completed') message('准备完成，可以开始复核。')
      else message('准备失败，详情见下方日志。', true)
    }
  }

  async function loadPipelinePreview() {
    pipelineError = ''
    try {
      pipelinePreview = await request<PipelinePreview>(`/api/pipeline/preview?days=${remoteDays === 'all' ? 365 : remoteDays}`)
    } catch (cause) {
      pipelinePreview = null
      pipelineError = cause instanceof Error ? cause.message : '无法读取 R2'
    }
  }

  async function startPipeline() {
    try {
      pipeline = await request<PipelineState>('/api/pipeline/prepare', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ days: remoteDays === 'all' ? 365 : Number(remoteDays), holdout_ratio: holdoutRatio }),
      })
      await refreshPipeline()
    } catch (cause) { message(cause instanceof Error ? cause.message : '启动失败', true) }
  }

  function setAutoPrepare(value: boolean) {
    autoPrepare = value
    try { localStorage.setItem(AUTO_PREPARE_KEY, value ? '1' : '0') } catch {}
  }

  async function reviewPipelineBatch() {
    await refreshBatches(pipeline?.batch_id || pipelinePreview?.latest_batch || undefined)
    if (!batch) return message('还没有批次。', true)
    openStep('review')
  }

  async function trainPipelineBatch() {
    await refreshBatches(pipeline?.batch_id || pipelinePreview?.latest_batch || undefined)
    if (!batch) return message('还没有批次。', true)
    trainingBackend = 'kaggle'
    active = 'training'
    await loadTrainingStep()
  }

  $: pipelineBatch = batches.find((item) => item.batch_id === (pipeline?.batch_id || pipelinePreview?.latest_batch)) || null
  $: pipelineStageIndex = Math.max(0, pipelineStages.findIndex(([name]) => name === pipeline?.stage))
  $: pipelineHeadline = pipelineRunning()
    ? '正在准备训练数据…'
    : pipeline?.status === 'failed' ? '准备失败'
    : pipelineBatch && pipelineBatch.review && pipelineBatch.review.pending === 0 && pipelineBatch.review.total > 0 ? '标签已就绪，可以训练'
    : pipelineBatch && pipelineBatch.review && pipelineBatch.review.pending > 0 ? `还有 ${pipelineBatch.review.pending} 条待复核`
    : pipelinePreview ? (pipelinePreview.new ? `发现 ${pipelinePreview.new} 张新截图` : '没有新截图')
    : '准备训练数据'

  onMount(async () => {
    try {
      try { autoPrepare = localStorage.getItem(AUTO_PREPARE_KEY) === '1' } catch {}
      await Promise.all([refreshBatches(), loadRemoteStatus(), request<{ configured: boolean }>('/api/jev/config').then((config) => { jevConfigured = config.configured }).catch(() => { jevConfigured = false })])
      request<{ build: string }>('/api/health').then((health) => { build = health.build }).catch(() => {})
      await refreshPipeline()
      if (remoteConfig?.configured && !pipelineRunning()) {
        await loadPipelinePreview()
        if (autoPrepare && pipelinePreview?.new) await startPipeline()
      }
    } catch { message('无法连接 Studio API。', true) }
  })
  onDestroy(() => {
    stopTrainingPoll()
    stopJevPoll()
    stopPipelinePoll()
    clearToastTimer()
  })
</script>

<svelte:window on:paste={pasteImages} on:keydown={onKeyDown} />

<svelte:head><meta name="description" content="OCRKit Studio" /></svelte:head>

<main class="app-shell">
  <div class="app-frame">
    <div class="app-chrome">
      <header class="app-header">
        <div class="brand">
          <h1>OCRKit Model Studio</h1>
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
              <option value={item.batch_id}>{item.batch_id}{item.screenshot_set ? ` · 集v${item.screenshot_set.version}` : ''}</option>
            {/each}
          </select>
          {#if batch}
            <span class="batch-revision" title={batch.active_dataset_revision || '当前批次尚未生成候选 revision'}>
              <span>当前 revision</span>
              <code>{batch.active_dataset_revision || '尚未生成'}</code>
            </span>
          {/if}
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
            <span class="nav-index">{item[1]}</span>
            <span class="nav-label">{item[2]}</span>
          </button>
        {/each}
      </nav>
    </div>

    {#if toastVisible && toastBody()}
      <div
        class="toast"
        class:toast-error={toastKind() === 'error'}
        class:toast-success={toastKind() === 'success'}
        role="status"
        aria-live={toastKind() === 'error' ? 'assertive' : 'polite'}
      >
        <span class="toast-mark" aria-hidden="true">{toastKind() === 'error' ? '!' : '✓'}</span>
        <div class="toast-copy">
          <strong>{toastTitle()}</strong>
          <p>{toastBody()}</p>
        </div>
        <button type="button" class="toast-dismiss" on:click={dismissToast} aria-label="关闭提示">关闭</button>
      </div>
    {/if}

    <div class="app-content">

    {#if active === 'import'}
      <section class="panel pipeline-panel" aria-label="一键准备">
        <header class="panel-head">
          <p class="eyebrow">快速开始</p>
          <h2>{pipelineHeadline}</h2>
          <p>{pipelineRunning() ? '可以离开此页，任务在后台继续。' : `自动选取最近 ${remoteDays === 'all' ? '全部' : remoteDays + ' 天'}尚未用过的 R2 截图 → 切片与 OCR → 双引擎一致的 Train 直接采纳，其余交 JEV；高置信结果自动成为 Train 标签，Holdout 与不确定项留给你复核。`}</p>
        </header>
        {#if pipelineError}<p class="remote-empty">{pipelineError}</p>{/if}
        {#if pipeline && pipeline.status !== 'not_started'}
          {#if pipelineRunning() || pipeline.status === 'failed'}
          <ol class="pipeline-stages">
            {#each pipelineStages as [name, label], index}
              <li class:stage-done={pipeline?.status === 'completed' || index < pipelineStageIndex} class:stage-current={pipelineRunning() && index === pipelineStageIndex}>{label}{#if pipelineRunning() && index === pipelineStageIndex && pipeline?.total}<span> {pipeline.completed ?? 0}/{pipeline.total}</span>{/if}</li>
            {/each}
          </ol>
          {/if}
          {#if pipeline?.status === 'completed'}
            <p class="status-meta">新截图 {pipeline.new_screenshots ?? 0} 张 · 双引擎一致自动采纳 {pipeline.agreement_adopted ?? 0}{#if pipeline.jev_adopted} · JEV 另采纳 接受 {pipeline.jev_adopted.accepted} / 拒绝 {pipeline.jev_adopted.rejected}{:else} · JEV 未运行{/if}{#if pipeline.jev_errors} · JEV 错误 {pipeline.jev_errors}{/if}</p>
          {/if}
          {#if pipeline?.status === 'failed' && pipeline.log_tail}<pre class="pipeline-log">{pipeline.log_tail.slice(-3000)}</pre>{/if}
        {/if}
        <div class="panel-actions">
          {#if pipelineBatch?.review && pipelineBatch.review.pending > 0 && !pipelineRunning()}
            <button class="button-primary" on:click={() => void reviewPipelineBatch()}>开始复核（{pipelineBatch.review.pending} 条）</button>
          {:else if pipelineBatch?.review && pipelineBatch.review.total > 0 && !pipelineRunning()}
            <button class="button-primary" disabled={busy} on:click={() => void trainPipelineBatch()}>选择训练方式</button>
          {/if}
          <button class={pipelineBatch?.review?.total && !pipelineRunning() ? 'button-secondary' : 'button-primary'} disabled={pipelineRunning() || !remoteConfig?.configured || (!!pipelinePreview && pipelinePreview.new === 0)} on:click={() => void startPipeline()}>
            {pipelineRunning() ? '准备中…' : pipelinePreview?.new ? `准备 ${pipelinePreview.new} 张新截图` : '准备新截图'}
          </button>
          <label class="field"><span>时间范围</span><select class="control" bind:value={remoteDays} disabled={pipelineRunning()} on:change={() => void loadPipelinePreview()}><option value="30">最近 30 天</option><option value="7">最近 7 天</option><option value="all">一年内</option></select></label>
          <label class="field field-check"><input type="checkbox" checked={autoPrepare} on:change={(event) => setAutoPrepare(event.currentTarget.checked)} /><span>打开时自动准备</span></label>
          {#if !jevConfigured}<span class="status-meta">JEV 未配置：跳过自动建议，全部候选留给你复核。</span>{/if}
        </div>
      </section>
      <details class="manual-import" on:toggle={(event) => { if (event.currentTarget.open && remoteConfig?.configured && !remoteObjects.length) void loadRemoteImages() }}>
      <summary>手动选择 / 上传截图（高级）</summary>
      <section class="panel-grid">
        <div class="panel">
          <header class="panel-head"><p class="eyebrow">步骤 1</p><h2>选择截图，准备训练数据</h2><p>自动加载最近 30 天 R2 截图，选图后创建批次并切片，使用 JEV 建议辅助确认标签。</p></header>
          <section class="remote-import" aria-label="R2 远程截图">
            <header class="remote-head">
              <div>
                <p class="eyebrow">默认数据源</p>
                <h3>从 R2 获取截图</h3>
                <p>直接读取 R2，无需平台登录或截图集 token。使用本机已有 Cloudflare 登录读取 D1 中的准确反馈。反馈帮助选图，仍需确认切片文字作为训练标签。</p>
              </div>
              {#if remoteConfig?.configured}
                <span class="status-pill status-done">只读 · {remoteConfig.bucket}</span>
              {/if}
            </header>
            {#if !remoteConfig?.configured}
              <p class="remote-empty">未配置 Studio R2。设置 <code>OCRKIT_STUDIO_R2_BUCKET</code> 与 <code>OCRKIT_STUDIO_R2_ALLOWED_PREFIXES</code> 后重启 Studio。</p>
            {:else}
              <p>{feedbackLoading ? '正在读取平台准确反馈…' : feedbackAvailable ? '平台准确反馈已读取，可筛选识别不准确截图。' : '暂未读取平台反馈，可继续选图与训练。'}</p>
              <div class="remote-controls">
                <label class="field">
                  <span>Prefix</span>
                  <input class="control" bind:value={remotePrefix} placeholder={remoteConfig.allowed_prefixes[0]} />
                </label>
                <label class="field"><span>时间范围</span><select class="control" bind:value={remoteDays} disabled={remoteLoading} on:change={() => void loadRemoteImages()}><option value="30">最近 30 天</option><option value="7">最近 7 天</option><option value="all">全部</option></select></label>
                <button class="button-secondary" disabled={remoteLoading || busy} on:click={() => void loadRemoteImages()}>
                  {remoteLoading ? `加载中（${remoteObjects.length} 张）…` : '刷新截图'}
                </button>
              </div>

              {#if remoteObjects.length}
                <div class="remote-filter-bar">
                  <div class="remote-search-box">
                    <svg class="search-icon" viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                      <path fill-rule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clip-rule="evenodd" />
                    </svg>
                    <input
                      type="text"
                      class="remote-search-input"
                      placeholder="搜索文件名、路径或上传日期..."
                      bind:value={remoteFilterText}
                    />
                    {#if remoteFilterText}
                      <button type="button" class="clear-search-btn" on:click={() => (remoteFilterText = '')} aria-label="清除搜索">✕</button>
                    {/if}
                  </div>

                  <label class="field"><span>平台反馈</span><select class="control" bind:value={feedbackFilter}><option value="all">全部</option><option value="inaccurate">识别不准确</option><option value="accurate">识别准确</option><option value="unknown">无反馈</option></select></label>
                  <div class="remote-toolbar-group">
                    <div class="remote-select-group">
                      <button
                        type="button"
                        class="button-secondary button-compact"
                        on:click={toggleSelectAll}
                        title={isAllSelected ? '取消全选当前筛选列表' : '全选当前筛选列表'}
                      >
                        {isAllSelected ? '取消全选' : '全选'}
                      </button>
                      <button
                        type="button"
                        class="button-secondary button-compact"
                        on:click={invertSelection}
                        title="反选当前筛选列表"
                      >
                        反选
                      </button>
                      {#if remoteSelected.size > 0}
                        <button
                          type="button"
                          class="button-secondary button-compact"
                          on:click={clearSelection}
                          title="清空所有已选"
                        >
                          清空
                        </button>
                      {/if}
                    </div>

                    <div class="remote-sort-group">
                      <label class="remote-sort-label">
                        <span>排序</span>
                        <select bind:value={remoteSortField} class="remote-sort-select">
                          <option value="date">上传日期</option>
                          <option value="size">文件大小</option>
                          <option value="name">文件名</option>
                        </select>
                      </label>
                      <button
                        type="button"
                        class="button-secondary button-compact"
                        on:click={() => (remoteSortOrder = remoteSortOrder === 'desc' ? 'asc' : 'desc')}
                        title={remoteSortOrder === 'desc' ? '当前降序 (点击切换升序)' : '当前升序 (点击切换降序)'}
                      >
                        {remoteSortOrder === 'desc' ? '↓ 降序' : '↑ 升序'}
                      </button>
                    </div>

                    <div class="segmented" role="group" aria-label="视图切换">
                      <button
                        type="button"
                        class="segmented-btn"
                        class:segmented-active={remoteViewMode === 'list'}
                        on:click={() => (remoteViewMode = 'list')}
                        title="列表视图"
                      >
                        列表
                      </button>
                      <button
                        type="button"
                        class="segmented-btn"
                        class:segmented-active={remoteViewMode === 'grid'}
                        on:click={() => (remoteViewMode = 'grid')}
                        title="网格视图"
                      >
                        网格
                      </button>
                    </div>
                  </div>
                </div>

                <div class="remote-list-header">
                  <label class="remote-master-select">
                    <input
                      type="checkbox"
                      checked={isAllSelected}
                      indeterminate={isPartiallySelected}
                      on:change={toggleSelectAll}
                    />
                    <span>
                      已选 <strong>{remoteSelected.size}</strong> 项
                      {#if remoteSelected.size > 0}
                        <small class="remote-selected-bytes">({formatBytes(selectedTotalBytes)})</small>
                      {/if}
                    </span>
                  </label>
                  <span class="remote-count-meta">
                    显示 {filteredRemoteObjects.length} / 共 {remoteObjects.length} 张截图
                  </span>
                </div>

                {#if filteredRemoteObjects.length === 0}
                  <div class="remote-no-match">
                    <p>没有匹配 "{remoteFilterText}" 的候选截图</p>
                    <button type="button" class="button-secondary button-compact" on:click={() => (remoteFilterText = '')}>清除筛选条件</button>
                  </div>
                {:else if remoteViewMode === 'list'}
                  <div class="remote-list" role="list">
                    {#each filteredRemoteObjects as object (object.key)}
                      {@const keyInfo = splitKey(object.key)}
                      <div
                        class="remote-row"
                        class:remote-row-selected={remoteSelected.has(object.key)}
                        role="listitem"
                      >
                        <label class="remote-row-check">
                          <input
                            type="checkbox"
                            checked={remoteSelected.has(object.key)}
                            on:change={() => toggleRemoteSelection(object.key)}
                          />
                        </label>

                        <button
                          type="button"
                          class="remote-thumb-wrapper"
                          on:mouseenter={() => showHoverPreview(object)}
                          on:mouseleave={hideHoverPreview}
                          on:click={() => openPreviewModal(object)}
                          aria-label={`查看 ${keyInfo.name} 大图预览`}
                        >
                          <img
                            src={`/api/r2/image?key=${encodeURIComponent(object.key)}`}
                            alt={keyInfo.name}
                            loading="lazy"
                            class="remote-thumb-img"
                          />
                          {#if keyInfo.ext}
                            <span class="remote-thumb-badge">{keyInfo.ext}</span>
                          {/if}
                        </button>

                        <div class="remote-row-info">
                          <div class="remote-row-primary">
                            <strong class="remote-filename" title={object.key}>{keyInfo.name}</strong>
                            {#if keyInfo.dir}
                              <span class="remote-path" title={keyInfo.dir}>{keyInfo.dir}</span>
                            {/if}
                          </div>
                          <div class="remote-row-meta"><span>{feedbackLabel(object.key)}</span>
                            <span class="meta-item meta-date" title="上传时间">
                              <svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13">
                                <path fill-rule="evenodd" d="M5.75 2a.75.75 0 01.75.75V4h7V2.75a.75.75 0 011.5 0V4h.25A2.75 2.75 0 0118 6.75v8.5A2.75 2.75 0 0115.25 18H4.75A2.75 2.75 0 012 15.25v-8.5A2.75 2.75 0 014.75 4H5V2.75A.75.75 0 015.75 2zm-1 5.5c-.69 0-1.25.56-1.25 1.25v6.5c0 .69.56 1.25 1.25 1.25h10.5c.69 0 1.25-.56 1.25-1.25v-6.5c0-.69-.56-1.25-1.25-1.25H4.75z" clip-rule="evenodd" />
                              </svg>
                              {formatDate(object.last_modified)}
                            </span>
                            <span class="meta-item meta-size">
                              {formatBytes(object.size)}
                            </span>
                          </div>
                        </div>

                        <div class="remote-row-actions">
                          <button
                            type="button"
                            class="button-preview-trigger"
                            on:click={() => openPreviewModal(object)}
                            title="放大预览"
                          >
                            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                              <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                              <path fill-rule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" />
                            </svg>
                            <span>预览</span>
                          </button>
                        </div>
                      </div>
                    {/each}
                  </div>
                {:else}
                  <div class="remote-grid" role="list">
                    {#each filteredRemoteObjects as object (object.key)}
                      {@const keyInfo = splitKey(object.key)}
                      <div
                        class="remote-card"
                        class:remote-card-selected={remoteSelected.has(object.key)}
                        role="listitem"
                      >
                        <div class="remote-card-media">
                          <img
                            src={`/api/r2/image?key=${encodeURIComponent(object.key)}`}
                            alt={keyInfo.name}
                            loading="lazy"
                            class="remote-card-img"
                            on:mouseenter={() => showHoverPreview(object)}
                            on:mouseleave={hideHoverPreview}
                          />
                          <div class="remote-card-overlay">
                            <label class="remote-card-check">
                              <input
                                type="checkbox"
                                checked={remoteSelected.has(object.key)}
                                on:change={() => toggleRemoteSelection(object.key)}
                              />
                            </label>
                            <span class="remote-card-size">{formatBytes(object.size)}</span>
                          </div>
                          <button
                            type="button"
                            class="remote-card-expand"
                            on:click={() => openPreviewModal(object)}
                            title="放大预览"
                          >
                            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                              <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                              <path fill-rule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" />
                            </svg>
                          </button>
                        </div>
                        <div class="remote-card-content">
                          <strong class="remote-card-title" title={object.key}>{keyInfo.name}</strong>
                          <span class="remote-card-date">{formatDate(object.last_modified)}</span><span class="status-tag">{feedbackLabel(object.key)}</span>
                        </div>
                      </div>
                    {/each}
                  </div>
                {/if}

                <div class="remote-actions">
                  <div class="remote-selection-summary">
                    <span>已选 <strong>{remoteSelected.size}</strong> / {remoteObjects.length} 张截图</span>
                    {#if remoteSelected.size > 0}
                      <span class="remote-size-pill">{formatBytes(selectedTotalBytes)}</span>
                    {/if}
                  </div>
                  <div class="remote-action-buttons">
                    <label class="field"><span>导入到</span><select class="control" bind:value={remoteImportMode} disabled={busy}><option value="new">创建新批次</option><option value="append" disabled={!batch}>加入当前批次{batch ? ` ${batch.batch_id}` : '（先选择批次）'}</option></select></label>
                    <label><input type="checkbox" bind:checked={generateAfterImport} disabled={busy} /> 导入后自动切片并生成候选</label>
                    <button class="button-primary" disabled={busy || remoteLoading || !remoteSelected.size || (remoteImportMode === 'append' && !batch)} on:click={() => void importRemoteImages()}>
                      {remoteImportMode === 'append' ? `加入当前批次 (${remoteSelected.size} 张)` : `创建批次 (${remoteSelected.size} 张)`}
                    </button>
                  </div>
                </div>
              {:else}
                <p class="remote-empty">{remoteLoading ? '正在扫描截图目录…' : '当前范围没有截图，可以扩大时间范围或刷新。'}</p>
              {/if}
            {/if}
          </section><details class="set-import"><summary>本地文件 / 粘贴截图</summary>          <label class="file-pick" class:file-pick-ready={files.length > 0}>
            <input
              type="file"
              multiple
              accept="image/png,image/jpeg,image/webp"
              on:change={(event) => appendFiles(Array.from((event.currentTarget as HTMLInputElement).files || []))}
            />
            <span class="file-pick-title">{files.length ? `已选 ${files.length} 张` : '选择截图'}</span>
            <span class="file-pick-sub">{files.length ? '可继续添加，或直接创建 / 加入批次' : '也可 ⌘V / Ctrl+V 粘贴'}</span>
          </label>
          <div class="panel-actions">
            <label class="field field-compact">
              <span>Holdout</span>
              <input type="number" min="0" max="0.5" step="0.05" bind:value={holdoutRatio} />
            </label>
            <button class="button-primary" disabled={busy || !files.length} on:click={importFiles}>
              {busy ? '创建中…' : '创建批次'}
            </button>
            <button class="button-secondary" disabled={busy || !files.length || !batch} on:click={addFilesToBatch}>
              {busy ? '处理中…' : '加入当前批次'}
            </button>
          </div>
</details><details class="set-import"><summary>可选：平台已定稿截图集</summary>          <section class="set-import" aria-label="平台截图集导入">
            <div class="set-import-form">
              <label class="field">
                <span>截图集版本</span>
                <input class="control" type="number" min="1" step="1" bind:value={setVersionInput} placeholder="例如 3" />
              </label>
              <button class="button-primary" disabled={setImportBusy || !Number.isInteger(setVersionInput) || (setVersionInput ?? 0) < 1} on:click={() => void importScreenshotSet()}>
                {setImportBusy ? '导入中…' : '导入截图集'}
              </button>
            </div>
            <p class="set-import-hint">需配置 <code>OCRKIT_SCREENSHOT_SET_BASE_URL</code>、<code>OCRKIT_SCREENSHOT_SET_TOKEN</code> 与 Studio R2 读取；同一终态集合只导入一次，导入中断可重试续传。</p>
          </section>
</details>
        </div>
        <aside class="panel panel-side">
          <h3>流程</h3>
          <ol class="flow-list">
            <li class="flow-current"><b>1</b><span><strong>准备</strong> 选图 + 切片 + JEV</span></li>
            <li><b>2</b><span><strong>候选</strong> RapidOCR + Vision</span></li>
            <li><b>3</b><span><strong>复核</strong> 接受 / 拒绝</span></li>
            <li><b>4</b><span><strong>标签</strong> train / holdout</span></li>
            <li><b>5</b><span><strong>训练</strong> Kaggle GPU / CPU Smoke</span></li>
          </ol>
          {#if batch}
            <dl class="batch-stats">
              {#if batch.screenshot_set}
                <div><dt>截图集</dt><dd>v{batch.screenshot_set.version}</dd></div>
              {/if}
              <div><dt>截图</dt><dd>{batch.sources}</dd></div>
              <div><dt>Train</dt><dd>{batch.train_sources}</dd></div>
              <div><dt>Holdout</dt><dd>{batch.holdout_sources}</dd></div>
              <div><dt>待复核</dt><dd>{batch.review?.pending ?? '—'}</dd></div>
            </dl>
            <div class="batch-revision-summary">
              <span>当前候选 revision</span>
              <code>{batch.active_dataset_revision || '尚未生成'}</code>
            </div>
          {/if}
        </aside>
      </section>
      </details>
    {:else if active === 'candidates'}
      <section class="panel stage-panel">
        <header class="panel-head">
          <p class="eyebrow">步骤 2</p>
          <h2>生成候选</h2>
          <p>按固定 ROI 切片，并用上一版模型、RapidOCR 与 Vision 写入复核清单。上一版模型只提供识别建议，不代表人工真值。</p>
        </header>
        <div class="panel-actions">
          <button class="button-primary" disabled={!batch || busy} on:click={candidates}>
            {busy ? '生成中…' : batch?.review?.total ? '打开候选' : '生成候选'}
          </button>
          <button class="button-secondary" disabled={!batch || busy || !batch?.review?.total} on:click={() => void refreshVision()}>
            {busy ? '补回中…' : '补回 Vision（保留标注）'}
          </button>
          <button class="button-secondary" disabled={!batch || busy || !batch?.review?.total} on:click={() => void refreshTeacher()}>
            {busy ? '补回中…' : '补回上一版模型（保留标注）'}
          </button>
          <button class="button-secondary" disabled={!batch || busy || !batch?.review?.total} on:click={openRecreateModal}>
            {busy ? '重建中…' : '按最新 ROI 重建候选（保留人工决定）'}
          </button>
        </div>
      </section>
    {:else if active === 'review'}
      <section class="review">
        <header class="review-bar">
          <div class="segmented" role="group" aria-label="数据集">
            {#each ['train', 'holdout'] as name}
              <button type="button" class="segmented-btn" class:segmented-active={split === name} aria-pressed={split === name} on:click={() => { split = name; void refreshReview() }}>{name === 'train' ? 'Train' : 'Holdout'}</button>
            {/each}
          </div>
          <select class="control" aria-label="状态" bind:value={status} on:change={() => refreshReview()}>
            <option value="pending">待复核</option>
            <option value="auto_accepted">自动接受（抽查）</option>
            <option value="accepted">已接受</option>
            <option value="rejected">已拒绝</option>
            <option value="all">全部</option>
          </select>
          <select class="control" aria-label="区域" bind:value={roiFilter} on:change={onRoiFilterChange}>
            <option value="all">全部区域</option>
            {#each Object.entries(ROI_INFO) as [key, info]}
              <option value={key}>{info.name} ({rows.filter((row) => row.roi === key).length})</option>
            {/each}
          </select>
          <div class="review-progress">
            <span><b>{batch?.review?.pending ?? 0}</b> 待复核 · 共 {batch?.review?.total ?? 0}</span>
            <div><i style={`width: ${batch?.review?.total ? Math.round(((batch.review.total - batch.review.pending) / batch.review.total) * 100) : 0}%`}></i></div>
          </div>
          {#if jevState?.status === 'reviewing'}<span class="review-chip">JEV 处理中 {jevState.processed ?? 0}</span>{/if}
          <details class="more">
            <summary aria-label="更多操作">⋯</summary>
            <div class="more-menu">
              <p>{#if !jevConfigured}JEV 未配置{:else if jevState && jevState.status !== 'not_started'}JEV {jevState.status} · 已处理 {jevState.processed ?? 0} · 错误 {jevState.errors ?? 0}{#if jevState.jev_adopted} · 自动采纳 {jevState.jev_adopted.accepted + jevState.jev_adopted.rejected}{/if}{:else}JEV 尚未运行{/if}</p>
              <button class="button-secondary button-compact" disabled={!batch || busy || !jevConfigured || jevState?.status === 'reviewing'} on:click={() => void startJev()}>继续 JEV</button>
              {#if split === 'train' && (batch?.review?.teacher_eligible ?? 0) > 0}
                <button class="button-secondary button-compact" disabled={busy} on:click={() => void acceptTeacherSuggestions()}>接受上一版模型一致项（{batch?.review?.teacher_eligible}）</button>
              {/if}
              <button class="button-secondary button-compact" on:click={() => refreshReview()}>刷新</button>
            </div>
          </details>
        </header>
        <div class="review-body">
          <aside class="review-queue" aria-label="待复核队列">
            {#each displayedRows as row}
              <button type="button" class="queue-item" class:queue-item-current={selected?.crop === row.crop} on:click={(event) => selectCandidate(row, event)}>
                <i style={`background: ${getRoiInfo(row.roi).dotColor}`}></i>
                <span>{row.transcription || suggestionFor(row).text || '无建议'}</span>
                {#if row.review_status === 'accepted'}<em>✓</em>{:else if row.review_status === 'rejected'}<em>✕</em>{/if}
              </button>
            {/each}
          </aside>
          <article class="review-card">
            {#if selected && batch}
              {@const info = getRoiInfo(selected.roi)}
              {@const pick = suggestionFor(selected)}
              {@const index = displayedRows.findIndex((row) => row.crop === selected?.crop)}
              <div class="card-meta">
                <span class="roi-chip"><i style={`background: ${info.dotColor}`}></i>{info.name}</span>
                {#if selected.accuracy_feedback === 'inaccurate'}<span class="tag tag-warn">平台标记不准确</span>{/if}
                {#if selected.auto_reject_reason}<span class="tag">自动排除</span>{:else if selected.auto_accept_reason}<span class="tag">自动接受</span>{/if}
                <span class="card-index">{index + 1} / {displayedRows.length}</span>
              </div>
              <button type="button" class="crop-stage" aria-label="点击切换放大" on:click={toggleCropZoom}>
                <img
                  src={`/api/batches/${batch.batch_id}/crop?split=${split}&crop=${encodeURIComponent(selected.crop)}`}
                  alt="待复核切片"
                  class="crop-image"
                  class:crop-image-crisp={cropDisplayScale() > 1.25}
                  style={cropImageStyle()}
                  on:load={onCropLoad}
                />
              </button>
              {#if selected.roi === 'achievement_panel'}
                <div class="segmented" role="group" aria-label="类别">
                  {#each CATEGORIES as [value, label]}
                    <button type="button" class="segmented-btn" class:segmented-active={selected.category === value} aria-pressed={selected.category === value} on:click={() => { if (selected) selected = { ...selected, category: value } }}>{label}</button>
                  {/each}
                </div>
              {/if}
              <div class={`suggest suggest-${pick.tone}`}>
                <span class="suggest-source">{pick.label}</span>
                <span>{pick.hint}</span>
              </div>
              <input
                class="transcription"
                type="text"
                aria-label="转写"
                autocomplete="off"
                autocapitalize="off"
                spellcheck="false"
                enterkeyhint="done"
                bind:this={transcriptionEl}
                bind:value={selected.transcription}
                on:keydown={onTranscriptionKey}
              />
              <div class="alts">
                {#each alternativesFor(selected, selected.transcription) as alt}
                  <button type="button" class="alt" on:click={() => { applyEngineText(alt.text); transcriptionEl?.focus() }}>{alt.text}<small>{alt.who}</small></button>
                {/each}
              </div>
              <footer class="card-actions">
                <button type="button" class="button-ghost" aria-label="上一条" title="上一条 (Alt + ↑)" disabled={displayedRows.length < 2} on:click={selectPrevCandidate}>‹</button>
                <button type="button" class="button-secondary" class:button-suggested={pick.reject} title="拒绝 (Alt + ⌫)" on:click={() => save('rejected')}>拒绝</button>
                <button type="button" class="button-primary" title="接受 (Enter)" on:click={() => save('accepted')}>接受</button>
                <button type="button" class="button-ghost" aria-label="下一条" title="下一条 (Alt + ↓)" disabled={displayedRows.length < 2} on:click={selectNextCandidate}>›</button>
              </footer>
            {:else if (batch?.review?.pending ?? 0) === 0 && (batch?.review?.total ?? 0) > 0 && status === 'pending'}
              <div class="review-done">
                <h2>复核完成</h2>
                <p>已接受 {batch?.review?.accepted} · 已拒绝 {batch?.review?.rejected}。可以直接训练。</p>
                <div class="panel-actions"><button class="button-primary" disabled={busy} on:click={() => void trainCurrentBatch()}>选择训练方式</button></div>
              </div>
            {:else}
              <div class="review-done"><h2>没有符合筛选的切片</h2><p>换一个数据集、状态或区域；若还没有候选，回到「准备」。</p></div>
            {/if}
          </article>
        </div>
      </section>
    {:else if active === 'dataset'}
      <section class="dataset-layout">
        <header class="panel panel-head dataset-intro">
          <p class="eyebrow">步骤 4</p>
          <h2>生成与导出标签</h2>
          <p>先校验并写出 recognition labels，再按需导出到私有 datasets。导出不会自动提交或推送。</p>
          <div class="dataset-readiness" aria-live="polite">
            <span class="status-pill" class:status-done={(batch?.review?.pending ?? 1) === 0 && (batch?.review?.total ?? 0) > 0} class:status-running={(batch?.review?.pending ?? 0) > 0}>
              {datasetReadyHint()}
            </span>
            {#if batch?.review}
              <span class="status-meta">已接受 {batch.review.accepted}</span>
              <span class="status-meta">待复核 {batch.review.pending}</span>
              <span class="status-meta">已拒绝 {batch.review.rejected}</span>
              <span class="status-meta">总计 {batch.review.total}</span>
            {/if}
          </div>
        </header>

        <div class="dataset-cards">
          <article class="panel action-card">
            <header>
              <p class="eyebrow">校验</p>
              <h3>生成 labels</h3>
              <p>要求所有候选已接受或拒绝，且 train / holdout 各至少一条已接受转写。</p>
            </header>
            <ul class="checklist">
              <li class:check-ok={(batch?.review?.pending ?? -1) === 0}>无待复核项</li>
              <li class:check-ok={(batch?.review?.accepted ?? 0) > 0}>至少一条已接受</li>
              <li class:check-ok={(batch?.train_sources ?? 0) > 0 && (batch?.holdout_sources ?? 0) > 0}>train / holdout 均有源图</li>
            </ul>
            {#if lastFinalize}
              <p class="result-chip">最近生成 · train {lastFinalize.train} / holdout {lastFinalize.holdout}</p>
            {/if}
            <div class="panel-actions">
              <button class="button-primary" disabled={!batch || busy} on:click={finalize}>
                {busy ? '生成中…' : '生成 labels'}
              </button>
              <button class="button-secondary" disabled={!batch || busy || !lastFinalize} on:click={() => { active = 'training'; void loadTrainingStep() }}>
                前往训练
              </button>
            </div>
          </article>

          <article class="panel action-card">
            <header>
              <p class="eyebrow">归档</p>
              <h3>导出私有数据集</h3>
              <p>复制到 <code>datasets/labeled/rec/studio/&lt;batch-id&gt;/</code>，作为不可变导出包。</p>
            </header>
            <ul class="checklist">
              <li>先完成 labels 校验</li>
              <li>导出路径按批次固定，已存在则拒绝覆盖</li>
              <li>不自动 git commit / push</li>
            </ul>
            {#if lastExportPath}
              <p class="result-chip" title={lastExportPath}>最近导出 · {lastExportPath.split('/').slice(-3).join('/')}</p>
            {/if}
            <div class="panel-actions">
              <button class="button-secondary" disabled={!batch || busy} on:click={exportDataset}>
                {busy ? '导出中…' : '导出到私有 datasets'}
              </button>
            </div>
          </article>
        </div>
      </section>
    {:else}
      <section class="training-panel">
        <div class="training-head">
          <header class="panel-head">
            <p class="eyebrow">步骤 5</p>
            <h2>Kaggle 模型训练</h2>
            <p>默认提交到 Kaggle GPU，查看日志，结束后自动下载模型。本地 CPU Smoke 用于快速验证训练脚本。</p>
          </header>
          <div class="panel-actions">
            <button class="button-secondary" disabled={!batch || busy} on:click={() => refreshTraining()}>立即刷新</button>
            <button class="button-secondary" disabled={!batch || busy || trainingIsRunning(training?.status)} on:click={() => void evaluateTrained()} title="对最近一次已训练出的检查点重新跑评测门槛，不会重新训练">只评测已有检查点</button>
          </div>
        </div>

        <div class="training-config">
          <div class="config-row">
            <label class="field">
              <span>训练环境</span>
              <select class="control" bind:value={trainingBackend} disabled={busy || trainingIsRunning(training?.status)}>
                <option value="kaggle">Kaggle GPU</option>
                <option value="smoke">本地 CPU Smoke</option>
              </select>
            </label>
            <label class="field field-compact">
              <span>目标 Epoch</span>
              <input class="control" type="number" min="1" max="100" bind:value={trainingEpochs} disabled={!batch || busy || trainingIsRunning(training?.status)} />
            </label>
          </div>
          {#if trainingBackend === 'smoke'}
            <label class="field">
              <span>恢复 checkpoint</span>
              <select class="control" bind:value={resumeCheckpoint} disabled={!batch || busy || trainingIsRunning(training?.status)}>
                <option value="">从 PP-OCRv6 预训练权重开始</option>
                {#each resumeCheckpoints as checkpoint}
                  <option value={checkpoint.path}>{checkpoint.name}</option>
                {/each}
              </select>
            </label>
            <p class="config-note">恢复到新的 run，目标 Epoch 须高于已完成进度。</p>
          {/if}
          {#if trainingBackend === 'kaggle' && trainingOptions}
            <label class="field">
              <span>基础权重</span>
              <select class="control" bind:value={baseWeights} disabled={busy || trainingIsRunning(training?.status)}>
                <option value="">官方 PP-OCRv6 预训练（从头训练）</option>
                {#each trainingOptions.base_weights as weights}<option value={weights.path}>接着训练：{weights.name}</option>{/each}
              </select>
            </label>
            <fieldset class="data-choice" disabled={busy || trainingIsRunning(training?.status)}>
              <legend>训练数据 <em>已选 {1 + includedBatches.size} 个批次</em></legend>
              <div class="data-list">
                <label class="data-row data-row-current">
                  <input type="checkbox" checked disabled />
                  <span class="data-id">{batch?.batch_id}</span>
                  <span class="data-count">当前批次</span>
                </label>
                {#each trainingOptions.data_batches as item}
                  <label class="data-row">
                    <input type="checkbox" checked={includedBatches.has(item.batch_id)} on:change={() => toggleIncludedBatch(item.batch_id)} />
                    <span class="data-id">{item.batch_id}</span>
                    <span class="data-count">train {item.train} · holdout {item.holdout}</span>
                  </label>
                {/each}
              </div>
            </fieldset>
          {/if}
          <div class="config-footer">
            <p class="config-note">
              {#if trainingBackend === 'kaggle'}
                {baseWeights ? '接着已有权重训练' : '从官方预训练权重重新训练'}，使用 {1 + includedBatches.size} 个批次的标签。同一张截图只会出现在训练或保留集之一。
              {:else}
                本地 CPU 快速验证训练脚本，不用于发布。
              {/if}
            </p>
            <button class="button-primary" disabled={!batch || busy || trainingIsRunning(training?.status)} on:click={startTraining}>
              {busy ? '启动中…' : trainingIsRunning(training?.status) ? '训练中…' : '启动训练'}
            </button>
          </div>
        </div>

        {#if !batch}
          <p class="empty">请先选择批次。</p>
        {:else if !training || training.status === 'not_started'}
          <div class="empty-card">
            <p class="eyebrow">就绪</p>
            <h3>尚未启动训练</h3>
            <p>选择基础权重、训练数据和 Epoch，再点击「启动训练」；标签会在启动时自动生成。</p>
          </div>
        {:else}
          <div class="training-status" aria-live="polite">
            <span
              class="status-pill"
              class:status-running={trainingIsRunning(training.status)}
              class:status-done={training.status === 'completed'}
              class:status-failed={training.status === 'failed' || training.status === 'completed_or_failed'}
            >{trainingStatusLabel(training.status)}</span>
            {#if training.backend}<span class="status-meta">{training.backend}</span>{/if}{#if training.checkpoint}<span class="status-meta">本地模型：{training.checkpoint}</span>{/if}
            {#if training.base}<span class="status-meta">基础权重：{training.base.kind === 'official' ? '官方预训练（从头）' : training.base.name}</span>{/if}
            {#if training.data}<span class="status-meta">训练数据：{training.data.batches.length} 个批次 · train {training.data.train_samples ?? '—'} / holdout {training.data.holdout_samples ?? '—'}</span>{/if}
            {#if training.field_accuracy != null}<span class="status-meta">固定截图准确率 {(training.field_accuracy * 100).toFixed(2)}%</span>{/if}
            {#if training.pid}<span class="status-meta">PID {training.pid}</span>{/if}
            {#if trainingPolling}<span class="status-meta status-live">自动刷新 · {TRAINING_POLL_MS / 1000}s</span>{/if}
            {#if trainingUpdatedAt}<span class="status-meta">更新于 {trainingUpdatedAt}</span>{/if}
          </div>

          {#if training.backend === 'evaluate' && training.status === 'failed' && training.verdict === 'failed'}
            <div class="empty-card override-card">
              <p class="eyebrow">与线上模型的成对评测未通过</p>
              <ul class="reason-list">{#each training.reasons ?? [] as reason}<li>{reason}</li>{/each}</ul>
              <label class="field"><span>放行备注（可选；未通过的原因会自动记入发布记录）</span>
                <textarea class="control" rows="3" bind:value={overrideReason} placeholder="例如：同一份测试数据下与线上打平，退步仅 1 个被遮挡的样本，且候选针对新版 HUD 训练"></textarea>
              </label>
              <div class="panel-actions"><button class="button-secondary" disabled={busy} on:click={() => void acceptOverride()}>按负责人决定放行</button></div>
            </div>
          {:else if training.override}
            <p class="config-note">已按负责人决定放行：{training.override.reason}</p>
          {/if}

          {#if training.command?.length}
            <div class="training-command">
              <span>命令</span>
              <code title={training.command.join(' ')}>{training.command.join(' ')}</code>
            </div>
          {/if}

          <section class="log-panel" aria-label="训练日志">
            <header class="log-toolbar">
              <span>训练日志</span>
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

        <section class="publish-panel" aria-label="R2 模型发布">
          <div class="publish-head">
            <header class="panel-head">
              <p class="eyebrow">发布</p>
              <h2>发布到 R2</h2>
              <p>训练完成后可启动；本地评测通过后继续发布。将重新评测、导出新的不可变版本、上传 R2 并下载校验，不覆盖历史模型。</p>
            </header>
            <div class="publish-status" aria-live="polite">
              <span
                class="status-pill"
                class:status-running={publicationIsActive()}
                class:status-done={publication?.status === 'completed'}
                class:status-failed={publication?.status === 'failed' || publication?.status === 'completed_or_failed'}
              >{trainingStatusLabel(publication?.status || 'not_started')}</span>
              {#if publication?.pid}<span class="status-meta">PID {publication.pid}</span>{/if}
            </div>
          </div>

          {#if training?.status !== 'completed'}
            <div class="empty-card">
              <p class="eyebrow">门槛</p>
              <h3>需要先完成训练</h3>
              <p>当前训练状态为「{trainingStatusLabel(training?.status || 'not_started')}」。发布按钮将在训练成功后启用。</p>
            </div>
          {:else}
            <div class="publish-steps" aria-hidden="true">
              <span>评测</span><i></i><span>导出</span><i></i><span>上传</span><i></i><span>校验</span>
            </div>
            <label class="confirm-row">
              <input type="checkbox" bind:checked={publishConfirmed} disabled={busy || publicationIsActive()} />
              <span>
                <strong>确认写入 R2</strong>
                <small>我理解这会创建新的模型版本前缀，并触发完整发布流水线。</small>
              </span>
            </label>
            <div class="panel-actions">
              <button class="button-secondary" disabled={!batch || busy} on:click={() => refreshPublication()}>刷新发布状态</button>
              <button class="button-primary" disabled={!canPublish()} on:click={startPublication}>
                {publicationIsActive() ? '发布中…' : '发布到 R2'}
              </button>
            </div>
            {#if !publishConfirmed && !publicationIsActive()}<p class="config-note">先勾选上方「确认写入 R2」，才能点击发布。发布的是候选版本，上线需要在发布后单独提升为稳定版。</p>{/if}
          {/if}

          {#if publication && publication.status !== 'not_started'}
            {#if publication.command?.length}
              <div class="training-command">
                <span>发布命令</span>
                <code title={publication.command.join(' ')}>{publication.command.join(' ')}</code>
              </div>
            {/if}
            <section class="log-panel log-panel-publish" aria-label="发布日志">
              <header class="log-toolbar">
                <span>发布日志</span>
                <div class="log-toolbar-actions">
                  <label class="log-follow">
                    <input type="checkbox" bind:checked={followPublishLog} on:change={() => void scrollLogToBottom()} />
                    跟随底部
                  </label>
                  {#if publication.log}
                    <span class="status-meta log-path" title={publication.log}>{publication.log.split('/').slice(-3).join('/')}</span>
                  {/if}
                </div>
              </header>
              <pre class="log-tail" bind:this={publishLogEl}>{publication.log_tail?.trimEnd() || '（暂无发布输出）'}</pre>
            </section>
          {/if}
        </section>
      </section>
    {/if}
    {#if active === 'training'}
    <section class="publish-panel release-decision-panel" aria-label="候选与 stable 决策">
      <div class="publish-head">
        <header class="panel-head">
          <p class="eyebrow">模型决策</p>
          <h2>候选 → stable</h2>
          <p>候选发布只写入 candidate channel；只有比较报告和远端 artifact 校验都通过后，显式操作才会更新 stable。</p>
        </header>
        <div class="panel-actions">
          <button class="button-secondary" disabled={releaseBusy} on:click={() => refreshReleaseComparison()}>比较候选与 stable</button>
        </div>
      </div>
      {#if !releaseComparison}
        <p class="empty">尚未读取模型 channel；配置 R2 后刷新比较。</p>
      {:else}
        <div class="training-command release-manifests">
          <span>候选</span><code>{releaseComparison.candidate.version || '未知'} · {releaseComparison.candidate.manifest_key || '无 manifest'}</code>
          <span>stable</span><code>{releaseComparison.stable.version || '未知'} · {releaseComparison.stable.manifest_key || '无 manifest'}</code>
        </div>
        {#if releaseComparison.comparison?.field_accuracy}
          {@const pct = (value?: number | null) => (value == null ? '—' : `${(value * 100).toFixed(2)}%`)}
          <p class="config-note">候选与线上模型在同一批固定截图、同一份代码下成对测量：固定截图 {pct(releaseComparison.comparison.field_accuracy.candidate)}（线上 {pct(releaseComparison.comparison.field_accuracy.stable)}）{#if releaseComparison.comparison.holdout_accuracy?.candidate != null}；保留集识别 {pct(releaseComparison.comparison.holdout_accuracy.candidate)}（线上 {pct(releaseComparison.comparison.holdout_accuracy.stable)}）{/if}。</p>
        {/if}
        {#if releaseComparison.eligible}
          <p class="status-meta status-live">候选证据完整，可执行显式晋级。</p>
        {:else}
          <ul class="config-note release-reasons">
            {#each releaseComparison.reasons as reason}<li>{reason}</li>{/each}
          </ul>
        {/if}
        <div class="panel-actions release-actions">
          <label class="confirm-row">
            <input type="checkbox" bind:checked={releaseConfirmed} disabled={!releaseComparison.eligible || releaseBusy} />
            <span><strong>确认晋级候选到 stable</strong><small>仅改变 stable channel 选择，不修改任何已发布 artifact。</small></span>
          </label>
          <button class="button-primary" disabled={!releaseComparison.eligible || !releaseConfirmed || releaseBusy} on:click={promoteStable}>显式晋级 stable</button>
        </div>
        <div class="panel-actions release-actions">
          <label class="field">
            <span>回滚到历史 verified manifest</span>
            <select class="control" bind:value={rollbackManifest} disabled={releaseBusy}>
              <option value="">选择历史 manifest</option>
              {#each (releaseComparison.stable.history || []) as entry}
                {#if entry.manifest_key}<option value={entry.manifest_key}>{entry.manifest_key}</option>{/if}
              {/each}
            </select>
          </label>
          <button class="button-secondary" disabled={!rollbackManifest || releaseBusy} on:click={rollbackStable}>回滚 stable</button>
        </div>
      {/if}
    </section>
    {/if}
    </div>
    {#if build}<p class="build-tag" title="当前页面所连接的 Studio 来自哪个分支和提交">{build}</p>{/if}
  </div>

  {#if hoverPreviewObject}
    {@const keyInfo = splitKey(hoverPreviewObject.key)}
    <div class="remote-hover-card" role="tooltip">
      <div class="hover-card-media">
        <img src={`/api/r2/image?key=${encodeURIComponent(hoverPreviewObject.key)}`} alt={keyInfo.name} />
      </div>
      <div class="hover-card-body">
        <strong class="hover-card-title">{keyInfo.name}</strong>
        <div class="hover-card-meta">
          <span>{formatDate(hoverPreviewObject.last_modified)}</span>
          <span class="hover-card-sep">·</span>
          <span>{formatBytes(hoverPreviewObject.size)}</span>
          {#if hoverPreviewObject.etag}
            <span class="hover-card-sep">·</span>
            <span class="hover-card-etag">{hoverPreviewObject.etag.replace(/"/g, '').slice(0, 8)}</span>
          {/if}
        </div>
      </div>
    </div>
  {/if}

  {#if previewModalObject}
    {@const keyInfo = splitKey(previewModalObject.key)}
    {@const isSelected = remoteSelected.has(previewModalObject.key)}
    <div class="modal-backdrop" role="dialog" aria-modal="true" aria-label="截图详情预览">
      <button type="button" class="modal-backdrop-dismiss" on:click={closePreviewModal} aria-label="关闭预览"></button>
      <div class="modal-dialog">
        <header class="modal-header">
          <div class="modal-title-wrap">
            <strong class="modal-filename" title={previewModalObject.key}>{keyInfo.name}</strong>
            <span class="modal-path">{previewModalObject.key}</span>
          </div>
          <div class="modal-actions">
            <button
              type="button"
              class="button-secondary button-compact"
              class:modal-selected={isSelected}
              on:click={() => previewModalObject && toggleRemoteSelection(previewModalObject.key)}
            >
              {isSelected ? '✓ 已选中' : '+ 选中此截图'}
            </button>
            <button type="button" class="modal-close-btn" on:click={closePreviewModal} title="关闭 (Esc)">✕</button>
          </div>
        </header>

        <div class="modal-body">
          <button type="button" class="modal-nav-btn modal-nav-prev" on:click={prevPreview} title="上一张 (←)">
            ‹
          </button>
          <div class="modal-image-wrap">
            <img
              src={`/api/r2/image?key=${encodeURIComponent(previewModalObject.key)}`}
              alt={keyInfo.name}
              class="modal-image"
            />
          </div>
          <button type="button" class="modal-nav-btn modal-nav-next" on:click={nextPreview} title="下一张 (→)">
            ›
          </button>
        </div>

        <footer class="modal-footer">
          <div class="modal-meta-grid">
            <div class="meta-cell">
              <span class="meta-label">上传时间</span>
              <strong class="meta-value">{formatDate(previewModalObject.last_modified)}</strong>
            </div>
            <div class="meta-cell">
              <span class="meta-label">文件大小</span>
              <strong class="meta-value">{formatBytes(previewModalObject.size)}</strong>
            </div>
            {#if previewModalObject.etag}
              <div class="meta-cell">
                <span class="meta-label">ETag</span>
                <span class="meta-value font-mono">{previewModalObject.etag.replace(/"/g, '')}</span>
              </div>
            {/if}
          </div>
          <span class="modal-keyboard-hint">快捷键：← / → 切换 · 空格 选择 · Esc 关闭</span>
        </footer>
      </div>
    </div>
  {/if}

  {#if showRecreateModal}
    <div class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="recreate-modal-title">
      <button
        type="button"
        class="modal-backdrop-dismiss"
        disabled={busy}
        on:click={closeRecreateModal}
        aria-label="关闭对话框"
      ></button>
      <div class="modal-dialog recreate-modal-dialog">
        <header class="recreate-modal-header">
          <div class="recreate-modal-title-wrap">
            <span class="recreate-modal-eyebrow">ROI 重建确认</span>
            <h3 id="recreate-modal-title">按最新 ROI 配置重新切片</h3>
          </div>
          <button
            type="button"
            class="modal-close-btn"
            disabled={busy}
            on:click={closeRecreateModal}
            aria-label="关闭"
          >×</button>
        </header>

        <div class="recreate-modal-body">
          <p class="recreate-modal-desc">
            将基于当前最新的 ROI 配置重新切片并提取候选文本。系统会自动继承已完成的审核，无需重复核对。
          </p>

          <div class="recreate-info-grid">
            <div class="recreate-info-card">
              <span class="recreate-card-title">人工决定智能继承</span>
              <p class="recreate-card-text">
                位置与尺寸完全吻合的切片，已接受或已拒绝的人工审核结论将自动迁移继承。
              </p>
            </div>
            <div class="recreate-info-card">
              <span class="recreate-card-title">未匹配项重新待复核</span>
              <p class="recreate-card-text">
                坐标发生变动或无法安全匹配的切片将重置为待复核，确保训练集质量。
              </p>
            </div>
            <div class="recreate-info-card">
              <span class="recreate-card-title">历史版本自动快照</span>
              <p class="recreate-card-text">
                当前的切片清单将安全归档为历史版本，随时可追溯，不发生不可逆覆写。
              </p>
            </div>
          </div>
        </div>

        <footer class="recreate-modal-footer">
          <button
            type="button"
            class="button-secondary button-compact"
            disabled={busy}
            on:click={closeRecreateModal}
          >
            取消
          </button>
          <button
            type="button"
            class="button-primary button-compact"
            disabled={busy}
            on:click={confirmRecreateCandidates}
          >
            {busy ? '正在重新切片…' : '确认重建候选'}
          </button>
        </footer>
      </div>
    </div>
  {/if}

  {#if downloadProgress}
    {@const percent = downloadProgress.total > 0 ? Math.min(100, Math.round((downloadProgress.completed / downloadProgress.total) * 100)) : 0}
    <div class="progress-backdrop" role="alertdialog" aria-modal="true" aria-label="下载与导入进度">
      <div class="progress-card">
        <div class="progress-header">
          <div class="progress-spinner" aria-hidden="true"></div>
          <div class="progress-title-wrap">
            <h4>{downloadProgress.stage === 'downloading' ? '正在并行下载 R2 截图…' : '正在处理批次数据…'}</h4>
            <p class="progress-subtitle">
              {downloadProgress.stage === 'downloading'
                ? `已完成 ${downloadProgress.completed} / ${downloadProgress.total} 张 (${percent}%)`
                : (downloadProgress.message || '正在解析切片与创建批次…')}
            </p>
          </div>
        </div>

        <div class="progress-track-wrapper">
          <div class="progress-track" role="progressbar" aria-valuenow={percent} aria-valuemin="0" aria-valuemax="100">
            <div class="progress-bar" style={`width: ${percent}%;`}></div>
          </div>
          <span class="progress-percent-label">{percent}%</span>
        </div>

        {#if downloadProgress.currentKey}
          <div class="progress-current-file">
            <span class="progress-file-label">最新完成:</span>
            <code class="progress-file-name" title={downloadProgress.currentKey}>
              {downloadProgress.currentKey.split('/').pop()}
            </code>
          </div>
        {/if}

        <div class="progress-foot">
          <span class="progress-hint">并发下载与切片解析中，请稍候</span>
        </div>
      </div>
    </div>
  {/if}
</main>
