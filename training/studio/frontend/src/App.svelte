<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte'

  type ReviewCounts = { total: number; accepted: number; pending: number; rejected: number; teacher_eligible?: number }
  type Batch = { batch_id: string; sources: number; train_sources: number; holdout_sources: number; quality_warnings: number; layout_version: string; review?: ReviewCounts }
  type Row = { crop: string; roi: string; review_status: string; auto_accept_reason?: string | null; auto_reject_reason?: string | null; candidate_text?: string; transcription?: string; suggested_transcription?: string | null; confidence?: number; rapidocr_text?: string; rapidocr_confidence?: number; vision_text?: string; vision_confidence?: number; teacher_model_version?: string | null; teacher_text?: string | null; teacher_confidence?: number | null; teacher_suggestion?: boolean; teacher_auto_accept_eligible?: boolean }
  type CropZoom = 'auto' | 1 | 2 | 3 | 4
  type TrainingState = {
    status?: string
    pid?: number
    log?: string
    log_tail?: string
    command?: string[]
  }
  type ResumeCheckpoint = { path: string; name: string }
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
  type RoiMeta = {
    name: string
    shortName: string
    code: string
    position: string
    dotColor: string
    rect: { x: number; y: number; w: number; h: number }
  }

  const ROI_METAS: Record<string, RoiMeta> = {
    left_panel: {
      name: '左侧统计与进度',
      shortName: '左侧统计',
      code: 'left_panel',
      position: '屏幕左侧',
      dotColor: '#0284c7',
      rect: { x: 2.7, y: 2.1, w: 14.5, h: 63.2 },
    },
    achievement_panel: {
      name: '左上成就与称号',
      shortName: '左上成就',
      code: 'achievement_panel',
      position: '屏幕左上角',
      dotColor: '#b45309',
      rect: { x: 2.7, y: 1.4, w: 14.5, h: 18.8 },
    },
    run_code_panel: {
      name: '房间运行代码',
      shortName: '运行代码',
      code: 'run_code_panel',
      position: '屏幕左中部',
      dotColor: '#6d28d9',
      rect: { x: 2.3, y: 26.4, w: 25.8, h: 13.2 },
    },
    center_banner: {
      name: '中央通关横幅',
      shortName: '中央横幅',
      code: 'center_banner',
      position: '屏幕正中央',
      dotColor: '#15803d',
      rect: { x: 11.7, y: 17.4, w: 78.9, h: 12.5 },
    },
    right_panel: {
      name: '右上地图与难度',
      shortName: '右上地图',
      code: 'right_panel',
      position: '屏幕右上角',
      dotColor: '#b91c1c',
      rect: { x: 81.3, y: 0, w: 18.4, h: 14.6 },
    },
    bottom_left_hero: {
      name: '左下英雄状态',
      shortName: '左下英雄',
      code: 'bottom_left_hero',
      position: '屏幕左下角',
      dotColor: '#4338ca',
      rect: { x: 2.3, y: 79.9, w: 23.4, h: 13.2 },
    },
  }

  function getRoiInfo(code: string): RoiMeta {
    return ROI_METAS[code] || {
      name: code,
      shortName: code,
      code,
      position: '切片区域',
      dotColor: '#52525b',
      rect: { x: 30, y: 30, w: 40, h: 40 },
    }
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
  let publishConfirmed = false
  let resumeCheckpoints: ResumeCheckpoint[] = []
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
  let remoteViewMode: RemoteViewMode = 'list'
  let hoverPreviewObject: RemoteObject | null = null
  let hoverPreviewTimeout: ReturnType<typeof setTimeout> | null = null
  let previewModalObject: RemoteObject | null = null
  let downloadProgress: DownloadProgress | null = null
  let cropZoom: CropZoom = 'auto'
  let cropNatural = { w: 0, h: 0 }
  let trainingPollTimer: ReturnType<typeof setInterval> | null = null
  let toastTimer: ReturnType<typeof setTimeout> | null = null
  let lastFinalize: { train: number; holdout: number } | null = null
  let lastExportPath = ''
  const cropZoomSteps: CropZoom[] = ['auto', 1, 2, 3, 4]
  const TRAINING_POLL_MS = 2000
  const TOAST_MS = 4200
  const supportedClipboardTypes = new Set(['image/png', 'image/jpeg', 'image/webp'])

  $: displayedRows = roiFilter === 'all' ? rows : rows.filter((row) => row.roi === roiFilter)

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
    batch = batches.find((item) => item.batch_id === (selectId || batch?.batch_id)) || batches[0] || null
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

  async function loadRemoteImages(append = false) {
    if (!remoteConfig?.configured) return message('Studio 尚未配置 R2 远程数据源。', true)
    if (!remotePrefix) return message('请输入或选择一个 R2 prefix。', true)
    remoteLoading = true
    try {
      const cursor = append ? remoteCursor : null
      const query = new URLSearchParams({ prefix: remotePrefix })
      if (cursor) query.set('cursor', cursor)
      const result = await request<{ objects: RemoteObject[]; next_cursor: string | null }>(`/api/r2/images?${query.toString()}`)
      remoteObjects = append ? [...remoteObjects, ...result.objects] : result.objects
      remoteCursor = result.next_cursor
      if (!append) remoteSelected = new Set()
      message(`已加载 ${result.objects.length} 张可用远程截图。`)
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
    const addingToExistingBatch = Boolean(batch)
    const totalKeys = remoteSelected.size
    busy = true
    downloadProgress = {
      stage: 'downloading',
      completed: 0,
      total: totalKeys,
      message: `准备并行下载 ${totalKeys} 张截图...`,
    }

    try {
      const endpoint = batch ? `/api/batches/${batch.batch_id}/remote-sources/stream` : '/api/batches/r2/stream'
      let finalResult: any = null

      await streamImport(endpoint, { keys: [...remoteSelected], holdout_ratio: holdoutRatio }, (chunk) => {
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
      await refreshReview()
      if (rows[0]) selectCandidate(rows[0])
    } catch (cause) { message(cause instanceof Error ? cause.message : '候选生成失败', true) } finally { busy = false }
  }

  async function acceptTeacherSuggestions() {
    if (!batch || split !== 'train' || !(batch.review?.teacher_eligible ?? 0)) return
    busy = true
    try {
      const result = await request<{ result: { accepted: number }; counts: ReviewCounts }>(`/api/batches/${batch.batch_id}/review/accept-teacher`, { method: 'POST' })
      batch = { ...batch, review: result.counts }
      await refreshReview()
      if (rows[0]) selectCandidate(rows[0])
      message(`已批量接受 ${result.result.accepted} 条上一版模型 + RapidOCR 一致项；holdout 未自动接受。`)
    } catch (cause) {
      message(cause instanceof Error ? cause.message : '批量接受上一版模型建议失败', true)
    } finally {
      busy = false
    }
  }

  function confidenceLabel(value?: number | null) {
    if (value == null || Number.isNaN(value)) return '—'
    return `${Math.round(value * 100)}%`
  }

  function selectCandidate(row: Row, event?: MouseEvent) {
    selected = { ...row, transcription: row.transcription || row.suggested_transcription || row.candidate_text || '' }
    cropZoom = 'auto'
    cropNatural = { w: 0, h: 0 }
    // Keep focus scroll inside the list pane; avoid jumping the whole page.
    const target = event?.currentTarget
    if (target instanceof HTMLElement) {
      target.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    }
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
      const currentIndex = displayedRows.findIndex((row) => row.crop === selected?.crop)
      // Prefer the items after the current one so review advances forward.
      const followingCrops = currentIndex >= 0
        ? displayedRows.slice(currentIndex + 1).map((row) => row.crop)
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

  async function refreshVision() {
    if (!batch) return message('先选择或创建批次。', true)
    busy = true
    try {
      const result = await request<{ review: ReviewCounts; summary: { rows: number; vision_covered: number; auto_accepted: number; auto_rejected: number; preserved_accepted: number; preserved_rejected: number } }>(`/api/batches/${batch.batch_id}/candidates/refresh-vision`, { method: 'POST' })
      batch = { ...batch, review: result.review }
      message(`已补回 Vision：${result.summary.vision_covered}/${result.summary.rows} 条有结果，自动排除 ${result.summary.auto_rejected} 条位置不匹配项；保留 ${result.summary.preserved_accepted} 条人工接受和 ${result.summary.preserved_rejected} 条人工拒绝。`)
      active = 'review'
      await refreshReview()
      if (rows[0]) selectCandidate(rows[0])
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
      await refreshReview()
      if (rows[0]) selectCandidate(rows[0])
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
    await Promise.all([refreshTraining(), refreshResumeCheckpoints(), refreshPublication()])
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
      }
      if (followPublishLog) await scrollLogToBottom()
    } catch (cause) {
      if (!options?.silent) message(cause instanceof Error ? cause.message : '获取发布状态失败', true)
    }
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

  async function startTraining() {
    if (!batch) return message('先选择批次。', true)
    busy = true
    try {
      await request<TrainingState>(`/api/batches/${batch.batch_id}/training/smoke`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ resume_checkpoint: resumeCheckpoint || null, epochs: trainingEpochs }),
      })
      message(resumeCheckpoint ? '已从所选 checkpoint 恢复训练，状态将自动刷新。' : 'Smoke 训练已启动，状态将自动刷新。')
      followLog = true
      await refreshTraining()
      startTrainingPoll()
    } catch (cause) { message(cause instanceof Error ? cause.message : '训练启动失败', true) } finally { busy = false }
  }

  onMount(async () => {
    try {
      await Promise.all([refreshBatches(), loadRemoteStatus()])
    } catch { message('无法连接 Studio API。', true) }
  })
  onDestroy(() => {
    stopTrainingPoll()
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
      <section class="panel-grid">
        <div class="panel">
          <header class="panel-head">
            <p class="eyebrow">步骤 1</p>
            <h2>导入截图</h2>
            <p>选择或粘贴 PNG / JPEG / WebP。可创建新批次，或补充到当前批次；重复内容会自动去重。</p>
          </header>
          <label class="file-pick" class:file-pick-ready={files.length > 0}>
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
          <section class="remote-import" aria-label="R2 远程截图">
            <header class="remote-head">
              <div>
                <p class="eyebrow">线上数据源</p>
                <h3>从 R2 获取截图</h3>
                <p>凭据只在 Studio 后端使用。远程截图会复制到私有 batch，并与本地截图一起进入候选复核。</p>
              </div>
              {#if remoteConfig?.configured}
                <span class="status-pill status-done">只读 · {remoteConfig.bucket}</span>
              {/if}
            </header>
            {#if !remoteConfig?.configured}
              <p class="remote-empty">未配置 Studio R2。设置 <code>OCRKIT_STUDIO_R2_BUCKET</code> 与 <code>OCRKIT_STUDIO_R2_ALLOWED_PREFIXES</code> 后重启 Studio。</p>
            {:else}
              <div class="remote-controls">
                <label class="field">
                  <span>Prefix</span>
                  <input class="control" bind:value={remotePrefix} placeholder={remoteConfig.allowed_prefixes[0]} />
                </label>
                <button class="button-secondary" disabled={remoteLoading || busy} on:click={() => void loadRemoteImages()}>
                  {remoteLoading ? '读取中…' : '加载对象'}
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
                          <div class="remote-row-meta">
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
                          <span class="remote-card-date">{formatDate(object.last_modified)}</span>
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
                    {#if remoteCursor}
                      <button class="button-secondary button-compact" disabled={remoteLoading || busy} on:click={() => void loadRemoteImages(true)}>
                        {remoteLoading ? '加载中…' : '加载下一页'}
                      </button>
                    {/if}
                    <button class="button-primary" disabled={busy || !remoteSelected.size} on:click={() => void importRemoteImages()}>
                      {batch ? `加入当前批次 (${remoteSelected.size} 张)` : `用所选截图创建批次 (${remoteSelected.size} 张)`}
                    </button>
                  </div>
                </div>
              {:else}
                <p class="remote-empty">输入允许的 prefix 后点击「加载对象」。只显示支持的图片格式和大小范围内的对象。</p>
              {/if}
            {/if}
          </section>
        </div>
        <aside class="panel panel-side">
          <h3>流程</h3>
          <ol class="flow-list">
            <li class="flow-current"><b>1</b><span><strong>导入</strong> 创建批次</span></li>
            <li><b>2</b><span><strong>候选</strong> RapidOCR + Vision</span></li>
            <li><b>3</b><span><strong>复核</strong> 接受 / 拒绝</span></li>
            <li><b>4</b><span><strong>标签</strong> train / holdout</span></li>
            <li><b>5</b><span><strong>训练</strong> CPU Smoke</span></li>
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
        </div>
      </section>
    {:else if active === 'review'}
      <section class="review-layout">
        <aside class="review-list">
          <div class="filters">
            <select class="control" bind:value={split} on:change={() => refreshReview()}>
              <option value="train">Train</option>
              <option value="holdout">Holdout</option>
            </select>
            <select class="control" bind:value={status} on:change={() => refreshReview()}>
              <option value="pending">待复核</option>
              <option value="accepted">已接受</option>
              <option value="auto_accepted">自动接受</option>
              <option value="teacher_eligible">上一版模型建议</option>
              <option value="rejected">已拒绝</option>
              <option value="all">全部状态</option>
            </select>
            <select class="control col-span-2" bind:value={roiFilter} on:change={onRoiFilterChange}>
              <option value="all">全部区域 ({rows.length})</option>
              {#each Object.entries(ROI_METAS) as [key, meta]}
                {@const count = rows.filter((r) => r.roi === key).length}
                <option value={key}>{meta.shortName} ({count})</option>
              {/each}
            </select>
            <button class="button-secondary button-compact col-span-2" on:click={() => refreshReview()}>刷新</button>
            {#if split === 'train' && (batch?.review?.teacher_eligible ?? 0) > 0}
              <button class="button-primary button-compact col-span-2" disabled={busy} on:click={() => void acceptTeacherSuggestions()}>
                接受上一版模型 + RapidOCR 一致项 ({batch?.review?.teacher_eligible})
              </button>
            {/if}
          </div>
          <div class="countline">
            <b>{displayedRows.length}</b>
            <span>条{roiFilter !== 'all' ? ` · ${getRoiInfo(roiFilter).shortName}` : ''}</span>
            <i>/</i>
            <b>{batch?.review?.pending || 0}</b>
            <span>待复核</span>
            <i>/</i>
            <b>{batch?.review?.total || 0}</b>
            <span>总计</span>
          </div>
          <div class="candidate-scroll">
            {#if displayedRows.length}
              {#each displayedRows as row}
                {@const roiInfo = getRoiInfo(row.roi)}
                <button
                  type="button"
                  class:selected-row={selected?.crop === row.crop}
                  class="candidate"
                  on:click={(event) => selectCandidate(row, event)}
                >
                  <div class="candidate-head">
                    <span class="candidate-roi-pill">
                      <span class="candidate-roi-dot" style={`background-color: ${roiInfo.dotColor};`}></span>
                      <span>{roiInfo.shortName}</span>
                    </span>
                    <span
                      class="candidate-conf-tag"
                      class:candidate-conf-high={(row.confidence ?? 0) >= 0.98}
                      class:candidate-conf-mid={(row.confidence ?? 0) >= 0.9 && (row.confidence ?? 0) < 0.98}
                    >
                      {confidenceLabel(row.confidence)}
                    </span>
                  </div>
                  <strong class="candidate-text">{row.candidate_text || '无候选文本'}</strong>
                  <small class="candidate-note">
                    {row.auto_reject_reason ? '自动排除 · 格式不匹配' : row.auto_accept_reason ? '自动接受 · 可抽查' : row.teacher_auto_accept_eligible ? '上一版模型建议' : (row.rapidocr_text && row.vision_text && row.rapidocr_text === row.vision_text ? '双引擎一致' : (row.rapidocr_text && row.vision_text ? '双引擎不一致' : '待人工核对'))}
                  </small>
                </button>
              {/each}
            {:else}
              <p class="empty">当前筛选无结果。若尚未生成候选，请回到上一步。</p>
            {/if}
          </div>
        </aside>
        <article class="review-detail">
          {#if selected && batch}
            {@const roiInfo = getRoiInfo(selected.roi)}
            {@const currentIndex = displayedRows.findIndex((r) => r.crop === selected?.crop)}
            <div class="review-detail-scroll">
              <header class="detail-header">
                <div class="detail-header-left">
                  <div class="roi-banner-pill">
                    <span class="roi-pill-dot" style={`background-color: ${roiInfo.dotColor};`}></span>
                    <strong class="roi-pill-name">{roiInfo.name}</strong>
                    <span class="roi-pill-code">{roiInfo.code}</span>
                    <span class="roi-pill-pos">{roiInfo.position}</span>
                  </div>
                  <div class="detail-subhead">
                    <h2>复核切片</h2>
                    <span
                      class="status-chip"
                      class:status-chip-pending={selected.review_status === 'pending'}
                      class:status-chip-accepted={selected.review_status === 'accepted' || selected.review_status === 'auto_accepted'}
                      class:status-chip-rejected={selected.review_status === 'rejected'}
                    >
                      {selected.auto_reject_reason ? '自动排除 · 格式不匹配' : selected.auto_accept_reason ? '自动接受 · 可人工覆盖' : selected.teacher_auto_accept_eligible ? '上一版模型建议 · 可接受' : selected.review_status === 'pending' ? '待复核' : selected.review_status === 'accepted' ? '已接受' : '已拒绝'}
                    </span>
                    <code class="crop-id-code" title={selected.crop}>{selected.crop.split('/').pop() || selected.crop}</code>
                  </div>
                </div>

                <div class="detail-header-right">
                  <div class="hud-minimap" title={`游戏画面方位：${roiInfo.position} (${roiInfo.name})`}>
                    <div class="hud-screen-frame">
                      <span class="hud-label">HUD 方位</span>
                      {#each Object.entries(ROI_METAS) as [key, meta]}
                        <div
                          class="hud-box"
                          class:hud-box-active={selected.roi === key}
                          style={`left: ${meta.rect.x}%; top: ${meta.rect.y}%; width: ${meta.rect.w}%; height: ${meta.rect.h}%; ${selected.roi === key ? `background-color: ${meta.dotColor}; border-color: ${meta.dotColor};` : ''}`}
                          title={`${meta.name} (${meta.position})`}
                        ></div>
                      {/each}
                    </div>
                  </div>

                  {#if cropNatural.w}
                    <p class="crop-meta" aria-live="polite">
                      {cropNatural.w}×{cropNatural.h}px · {cropZoom === 'auto' ? `自适应 ${cropDisplayScale().toFixed(1)}×` : `${cropZoom}×`}
                    </p>
                  {/if}
                </div>
              </header>

              <section class="crop-panel" aria-label="切片预览">
                <div class="crop-toolbar">
                  <div class="crop-toolbar-left">
                    <span class="crop-toolbar-label">切片预览</span>
                    <span class="crop-roi-tag">
                      <span class="crop-roi-dot" style={`background-color: ${roiInfo.dotColor};`}></span>
                      {roiInfo.shortName}
                    </span>
                  </div>
                  <div class="segmented" role="group" aria-label="切片缩放">
                    {#each cropZoomSteps as step}
                      <button
                        type="button"
                        class="segmented-btn"
                        class:segmented-active={cropZoom === step}
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
                <p class="engine-hint">点选引擎结果填入转写，确认后接受或拒绝。</p>
                {#if selected.auto_reject_reason}
                  <p class="auto-review-hint">这条切片未通过该 ROI 的内容格式检查，已自动排除，不会进入训练标签。若确认位置和内容都正确，可手动接受并填写转写。</p>
                {:else if selected.auto_accept_reason}
                  <p class="auto-review-hint">这条切片由自动规则接受，仍可修改转写或点击「拒绝」进行人工覆盖。</p>
                {:else if selected.teacher_auto_accept_eligible}
                  <p class="auto-review-hint">上一版模型与 RapidOCR 高置信度一致。这条建议只属于 Train，可用左侧按钮接受；Holdout 不会自动接受。</p>
                {/if}
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
                  <button
                    type="button"
                    class="engine-pick"
                    class:engine-pick-active={engineSelected(selected.teacher_text)}
                    class:engine-pick-empty={!selected.teacher_text}
                    disabled={!selected.teacher_text}
                    on:click={() => applyEngineText(selected?.teacher_text)}
                  >
                    <span class="engine-pick-head">
                      <span class="engine-name">上一版模型{selected.teacher_model_version ? ` · ${selected.teacher_model_version}` : ''}</span>
                      <span
                        class="engine-conf"
                        class:engine-conf-high={(selected.teacher_confidence ?? 0) >= 0.98}
                        class:engine-conf-mid={(selected.teacher_confidence ?? 0) >= 0.9 && (selected.teacher_confidence ?? 0) < 0.98}
                      >{confidenceLabel(selected.teacher_confidence)}</span>
                    </span>
                    <strong class="engine-text">{selected.teacher_text || '无识别结果'}</strong>
                    {#if engineSelected(selected.teacher_text)}<span class="engine-chosen">已选用</span>{/if}
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
              </div>
            </div>

            <footer class="review-actions-bar">
              <div class="review-actions-meta">
                {#if currentIndex >= 0}
                  <span class="review-index-badge">第 {currentIndex + 1} / {rows.length} 条</span>
                {/if}
                <div class="review-nav-group">
                  <button
                    type="button"
                    class="button-ghost button-compact"
                    disabled={!rows.length || busy}
                    on:click={selectPrevCandidate}
                    title="上一条 (Alt + ↑)"
                  >
                    ‹ 上一条
                  </button>
                  <button
                    type="button"
                    class="button-ghost button-compact"
                    disabled={!rows.length || busy}
                    on:click={selectNextCandidate}
                    title="下一条 (Alt + ↓)"
                  >
                    下一条 ›
                  </button>
                </div>
              </div>

              <div class="review-actions-main">
                <span class="review-shortcut-hint">⌘/Ctrl+Enter 接受 · Alt+⌫ 拒绝</span>
                <button
                  type="button"
                  class="button-secondary button-danger-hover"
                  disabled={busy}
                  on:click={() => save('rejected')}
                  title="拒绝此切片 (Alt + Backspace)"
                >
                  拒绝
                </button>
                <button
                  type="button"
                  class="button-primary"
                  disabled={busy}
                  on:click={() => save('accepted')}
                  title="接受此转写并切换到下一条 (Cmd/Ctrl + Enter)"
                >
                  接受并下一条
                </button>
              </div>
            </footer>
          {:else}
            <div class="empty-detail">
              <p class="eyebrow">步骤 3</p>
              <h2>从左侧选择一条候选</h2>
              <p>筛选 train / holdout 与状态，点条目后对照切片校正转写。</p>
            </div>
          {/if}
        </article>
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
            <h2>Smoke 训练</h2>
            <p>在当前批次 labels 上启动本地 CPU Smoke。失败 run 可从 checkpoint 恢复到新的 run。</p>
          </header>
          <div class="panel-actions">
            <button class="button-secondary" disabled={!batch || busy} on:click={() => refreshTraining()}>立即刷新</button>
            <button class="button-primary" disabled={!batch || busy || trainingIsRunning(training?.status)} on:click={startTraining}>
              {busy ? '启动中…' : trainingIsRunning(training?.status) ? '训练中…' : '启动训练'}
            </button>
          </div>
        </div>

        <div class="training-config">
          <label class="field">
            <span>恢复 checkpoint</span>
            <select class="control" bind:value={resumeCheckpoint} disabled={!batch || busy || trainingIsRunning(training?.status)}>
              <option value="">从 PP-OCRv6 预训练权重开始</option>
              {#each resumeCheckpoints as checkpoint}
                <option value={checkpoint.path}>{checkpoint.name}</option>
              {/each}
            </select>
          </label>
          <label class="field field-compact">
            <span>目标 Epoch</span>
            <input class="control" type="number" min="1" max="100" bind:value={trainingEpochs} disabled={!batch || busy || trainingIsRunning(training?.status)} />
          </label>
          <p class="config-note">恢复会带入模型、优化器与 epoch 状态，不覆盖原 run；目标 Epoch 须高于 checkpoint 已完成进度。</p>
        </div>

        {#if !batch}
          <p class="empty">请先选择批次。</p>
        {:else if !training || training.status === 'not_started'}
          <div class="empty-card">
            <p class="eyebrow">就绪</p>
            <h3>尚未启动训练</h3>
            <p>生成 labels 后配置 checkpoint 与 epoch，再点击「启动训练」。</p>
          </div>
        {:else}
          <div class="training-status" aria-live="polite">
            <span
              class="status-pill"
              class:status-running={trainingIsRunning(training.status)}
              class:status-done={training.status === 'completed'}
              class:status-failed={training.status === 'failed' || training.status === 'completed_or_failed'}
            >{trainingStatusLabel(training.status)}</span>
            {#if training.pid}<span class="status-meta">PID {training.pid}</span>{/if}
            {#if trainingPolling}<span class="status-meta status-live">自动刷新 · {TRAINING_POLL_MS / 1000}s</span>{/if}
            {#if trainingUpdatedAt}<span class="status-meta">更新于 {trainingUpdatedAt}</span>{/if}
          </div>

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
              <p>仅在 Smoke 成功后可用。将重新评测、导出新的不可变版本、上传 R2 并下载校验，不覆盖历史模型。</p>
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
              <h3>需要先完成 Smoke</h3>
              <p>当前训练状态为「{trainingStatusLabel(training?.status || 'not_started')}」。发布按钮将在 Smoke 成功后启用。</p>
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
    </div>
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
              on:click={() => toggleRemoteSelection(previewModalObject.key)}
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
