(function () {
  const fileSelect = document.getElementById('file-select');
  const customPathWrap = document.getElementById('custom-path-wrap');
  const customPath = document.getElementById('custom-path');
  const viewSelect = document.getElementById('view-select');
  const sampleInput = document.getElementById('sample-size');
  const keywordInput = document.getElementById('keyword-filter');
  const sortFieldSelect = document.getElementById('sort-field');
  const columnFocusSelect = document.getElementById('column-focus');
  const sortDescInput = document.getElementById('sort-desc');
  const loadBtn = document.getElementById('load-btn');
  const exportBtn = document.getElementById('export-csv');
  const downloadJsonBtn = document.getElementById('download-json');

  const loadStatus = document.getElementById('load-status');
  const sourceSummary = document.getElementById('source-summary');
  const datasetBadge = document.getElementById('dataset-badge');
  const metricGrid = document.getElementById('metric-grid');
  const alertList = document.getElementById('alert-list');
  const alertCount = document.getElementById('alert-count');
  const schemaSummary = document.getElementById('schema-summary');
  const schemaEl = document.getElementById('schema-content');
  const previewSummary = document.getElementById('preview-summary');
  const previewEl = document.getElementById('preview-table');
  const detailTitle = document.getElementById('detail-title');
  const detailEl = document.getElementById('record-detail');
  const rawEl = document.getElementById('json-raw');
  const tableWraps = Array.from(document.querySelectorAll('.inspector-table-wrap'));

  const state = {
    dataset: null,
    currentViewId: '',
    selectedIndex: -1,
    filteredRows: []
  };

  function computeFetchCandidates(path) {
    if (!path) {
      return [];
    }
    if (
      path.startsWith('http://') ||
      path.startsWith('https://') ||
      path.startsWith('/') ||
      path.startsWith('./') ||
      path.startsWith('../')
    ) {
      return [path];
    }

    const candidates = [];
    if (path.startsWith('frontend/')) {
      candidates.push(path.slice('frontend/'.length));
    }
    candidates.push('../' + path);
    candidates.push(path);
    return Array.from(new Set(candidates));
  }

  function escapeHtml(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function inferType(value) {
    if (value === null) {
      return 'null';
    }
    if (Array.isArray(value)) {
      return 'array';
    }
    if (typeof value === 'number') {
      return Number.isInteger(value) ? 'int' : 'float';
    }
    return typeof value;
  }

  function formatNumber(value) {
    if (value == null || value === '') {
      return '-';
    }
    if (typeof value === 'number') {
      return value.toLocaleString('zh-CN');
    }
    return String(value);
  }

  function formatPercent(value) {
    if (!Number.isFinite(value)) {
      return '-';
    }
    return value.toFixed(value >= 100 ? 0 : 1) + '%';
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) {
      return '-';
    }
    if (bytes === 0) {
      return '0 B';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let index = 0;
    while (value >= 1024 && index < units.length - 1) {
      value /= 1024;
      index += 1;
    }
    return value.toFixed(value >= 100 || index === 0 ? 0 : 1) + ' ' + units[index];
  }

  function stringifyValue(value) {
    if (value == null) {
      return '';
    }
    if (typeof value === 'object') {
      try {
        return JSON.stringify(value, null, 2);
      } catch (error) {
        return String(value);
      }
    }
    return String(value);
  }

  function valueForSearch(value) {
    if (value == null) {
      return '';
    }
    if (typeof value === 'object') {
      return stringifyValue(value).toLowerCase();
    }
    return String(value).toLowerCase();
  }

  function sanitizeRows(rows) {
    return rows.map(function (row, index) {
      if (row && typeof row === 'object' && !Array.isArray(row)) {
        const next = Object.assign({}, row);
        next.__row_id = index + 1;
        return next;
      }
      return {
        value: row,
        __row_id: index + 1
      };
    });
  }

  function summarizeRows(rows, sampleSize) {
    const sample = rows.slice(0, sampleSize);
    const keys = new Set();
    sample.forEach(function (row) {
      if (!row || typeof row !== 'object' || Array.isArray(row)) {
        return;
      }
      Object.keys(row).forEach(function (key) {
        if (key !== '__row_id') {
          keys.add(key);
        }
      });
    });

    const schema = Array.from(keys).sort().map(function (key) {
      let present = 0;
      const types = new Set();
      let example = null;
      const distinctValues = new Set();

      sample.forEach(function (row) {
        if (row && Object.prototype.hasOwnProperty.call(row, key)) {
          present += 1;
          const value = row[key];
          types.add(inferType(value));
          if (example === null && value !== undefined && value !== null && stringifyValue(value) !== '') {
            example = value;
          }
          distinctValues.add(stringifyValue(value));
        }
      });

      return {
        field: key,
        presentCount: present,
        presentPct: sample.length ? Math.round((present / sample.length) * 100) : 0,
        types: Array.from(types).sort(),
        example: example,
        distinctCount: distinctValues.size
      };
    });

    return {
      totalCount: rows.length,
      sampleCount: sample.length,
      schema: schema
    };
  }

  function toCsv(rows, fields) {
    function escapeCell(value) {
      if (value == null) {
        return '';
      }
      const text = typeof value === 'object' ? JSON.stringify(value) : String(value);
      if (/[",\n]/.test(text)) {
        return '"' + text.replace(/"/g, '""') + '"';
      }
      return text;
    }

    const header = fields.join(',');
    const body = rows.map(function (row) {
      return fields.map(function (field) {
        return escapeCell(row[field]);
      }).join(',');
    });
    return [header].concat(body).join('\n');
  }

  async function fetchText(path) {
    const candidates = computeFetchCandidates(path);
    const tried = [];

    for (let i = 0; i < candidates.length; i += 1) {
      const candidate = candidates[i];
      tried.push(candidate);
      const response = await fetch(candidate, { cache: 'no-store' });
      if (response.ok) {
        return response.text();
      }
    }

    throw new Error('HTTP 404 · tried: ' + tried.join(' | '));
  }

  async function fetchJsonMaybe(path) {
    const text = await fetchText(path);
    try {
      return JSON.parse(text);
    } catch (error) {
      return null;
    }
  }

  function parseStructuredText(text) {
    try {
      return JSON.parse(text);
    } catch (jsonError) {
      const lines = text.split(/\r?\n/).filter(Boolean);
      if (!lines.length) {
        throw new Error('文件为空');
      }
      const rows = lines.map(function (line) {
        return JSON.parse(line);
      });
      return rows;
    }
  }

  async function loadManifestDataset(path, manifest, sampleSize) {
    const basePath = path.slice(0, path.lastIndexOf('/') + 1);
    const files = manifest.files || {};

    async function loadOptionalRows(name, fallbackName, ndjsonName) {
      const samplePath = files[name] ? basePath + files[name] : basePath + fallbackName;
      const sampleJson = await fetchJsonMaybe(samplePath);
      if (Array.isArray(sampleJson)) {
        return sanitizeRows(sampleJson);
      }

      const ndPath = files[ndjsonName] ? basePath + files[ndjsonName] : '';
      if (!ndPath) {
        return [];
      }
      const text = await fetchText(ndPath);
      const rows = text.split(/\r?\n/).filter(Boolean).slice(0, sampleSize).map(function (line) {
        return JSON.parse(line);
      });
      return sanitizeRows(rows);
    }

    const booksRows = await loadOptionalRows('books_sample', 'books_sample.json', 'books_ndjson');
    const chapterRows = await loadOptionalRows('chapters_sample', 'chapters_sample.json', 'chapters_ndjson').catch(function () {
      return [];
    });

    const views = [];
    if (booksRows.length) {
      views.push({
        id: 'books',
        label: '书籍样本',
        description: '用于观察书籍级别字段完整性和样本结构。',
        rows: booksRows,
        focus: '样本',
        defaultSortField: 'name'
      });
    }
    if (chapterRows.length) {
      views.push({
        id: 'chapters',
        label: '章节样本',
        description: '用于观察章节正文、抓取时间和结构稳定性。',
        rows: chapterRows,
        focus: '章节',
        defaultSortField: 'book_id'
      });
    }

    if (!views.length) {
      views.push({
        id: 'manifest',
        label: 'Manifest 元信息',
        description: '样本文件不可用，仅展示 manifest 自身。',
        rows: sanitizeRows(Object.keys(manifest).map(function (key) {
          return { key: key, value: manifest[key] };
        })),
        focus: '元信息',
        defaultSortField: 'key'
      });
    }

    const metrics = [
      { label: '书籍总数', value: formatNumber(manifest.books_count || 0), detail: '主库导出统计' },
      { label: '章节总数', value: formatNumber(manifest.chapters_count || 0), detail: '导出时记录' },
      { label: '样本规模', value: formatNumber(manifest.sample_size || booksRows.length), detail: '每类导出样本数' },
      { label: '生成时间', value: manifest.generated_at || '-', detail: '最后导出时间' }
    ];

    const alerts = [];
    if (!booksRows.length) {
      alerts.push({ tone: 'warn', title: '未读取到 books 样本', detail: 'manifest 可用，但书籍样本未生成或路径不可访问。' });
    }
    if (!chapterRows.length) {
      alerts.push({ tone: 'info', title: '未读取到 chapters 样本', detail: '可以继续使用当前页面，但无法直接观察章节内容质量。' });
    }
    if ((manifest.chapters_count || 0) === 0) {
      alerts.push({ tone: 'danger', title: '章节总数为 0', detail: '导出成功但没有正文数据，通常意味着抓取流程未完成。' });
    }

    return {
      title: 'SQLite 导出观察',
      sourceType: 'manifest',
      sourceLabel: 'manifest.json',
      raw: manifest,
      metrics: metrics,
      alerts: alerts,
      views: views,
      sourceSummary: '来源 DB: ' + (manifest.source_db || manifest.source_db_path || '-')
    };
  }

  function loadShardDataset(indexPayload) {
    const shards = Array.isArray(indexPayload.shards) ? indexPayload.shards : [];
    const rows = sanitizeRows(shards.map(function (item) {
      const chapterCount = Number(item.source_chapter_count || item.chapter_count || 0);
      const fetchedCount = Number(item.source_fetched_chapter_count || item.source_book_fetched_chapters || 0);
      const missingCount = Math.max(0, chapterCount - fetchedCount);
      const coverageRate = chapterCount ? (fetchedCount / chapterCount) * 100 : 0;
      const sizeBytes = Number(item.size_bytes || 0);
      return Object.assign({}, item, {
        coverage_rate: Number(coverageRate.toFixed(2)),
        missing_chapters: missingCount,
        size_gb: Number((sizeBytes / (1024 * 1024 * 1024)).toFixed(2)),
        shard_range: String(item.book_id_start || '-').padStart(6, '0') + ' - ' + String(item.book_id_end || '-').padStart(6, '0')
      });
    }));

    const totalSourceChapters = rows.reduce(function (sum, row) {
      return sum + Number(row.source_chapter_count || row.chapter_count || 0);
    }, 0);
    const totalFetched = rows.reduce(function (sum, row) {
      return sum + Number(row.source_fetched_chapter_count || row.source_book_fetched_chapters || 0);
    }, 0);
    const zeroCoverageCount = rows.filter(function (row) {
      return Number(row.coverage_rate || 0) === 0;
    }).length;
    const partialCoverageCount = rows.filter(function (row) {
      const coverage = Number(row.coverage_rate || 0);
      return coverage > 0 && coverage < 100;
    }).length;
    const largestShard = rows.reduce(function (current, row) {
      if (!current || Number(row.size_bytes || 0) > Number(current.size_bytes || 0)) {
        return row;
      }
      return current;
    }, null);

    const alerts = [];
    if (zeroCoverageCount) {
      alerts.push({ tone: 'danger', title: zeroCoverageCount + ' 个分片正文覆盖率为 0%', detail: '这通常说明目录已抓到，但正文抓取尚未跑到这些 book_id 区间。' });
    }
    if (partialCoverageCount) {
      alerts.push({ tone: 'warn', title: partialCoverageCount + ' 个分片仍在补齐中', detail: '可以按 coverage_rate 排序，优先补抓缺口最大的分片。' });
    }
    if (largestShard) {
      alerts.push({ tone: 'info', title: '最大分片 ' + largestShard.file_name, detail: '当前体积 ' + formatBytes(Number(largestShard.size_bytes || 0)) + '，建议重点关注上传与校验耗时。' });
    }
    if (!alerts.length) {
      alerts.push({ tone: 'ok', title: '当前没有明显异常', detail: '所有分片都已形成可巡检数据。' });
    }

    return {
      title: 'Shard 巡检面板',
      sourceType: 'shards',
      sourceLabel: 'index.json',
      raw: indexPayload,
      metrics: [
        { label: '分片数', value: formatNumber(indexPayload.shard_count || rows.length), detail: '当前导出 shard 数量' },
        { label: '书籍总数', value: formatNumber(indexPayload.total_books || 0), detail: '分片聚合后的 book 数量' },
        { label: '章节覆盖率', value: formatPercent(totalSourceChapters ? (totalFetched / totalSourceChapters) * 100 : 0), detail: '正文抓取覆盖率' },
        { label: '最大分片', value: largestShard ? formatBytes(Number(largestShard.size_bytes || 0)) : '-', detail: largestShard ? largestShard.file_name : '无' }
      ],
      alerts: alerts,
      views: [
        {
          id: 'shards',
          label: '分片明细',
          description: '重点看 coverage_rate、missing_chapters、size_gb。',
          rows: rows,
          focus: '分片',
          defaultSortField: 'coverage_rate'
        }
      ],
      sourceSummary: '主库: ' + (indexPayload.source_db_path || '-') + ' · 输出目录: ' + (indexPayload.output_dir || '-')
    };
  }

  function loadGenericDataset(payload, label) {
    if (Array.isArray(payload)) {
      return {
        title: '通用数组观察',
        sourceType: 'array',
        sourceLabel: label,
        raw: payload,
        metrics: [
          { label: '记录数', value: formatNumber(payload.length), detail: '当前数组元素数量' }
        ],
        alerts: [
          { tone: 'info', title: '通用模式', detail: '当前数据不是专用 manifest/shards 结构，按通用数组方式展示。' }
        ],
        views: [
          {
            id: 'rows',
            label: '数组记录',
            description: '支持关键词过滤、字段聚焦和 CSV 导出。',
            rows: sanitizeRows(payload),
            focus: '记录',
            defaultSortField: ''
          }
        ],
        sourceSummary: '已加载通用数组数据。'
      };
    }

    if (payload && typeof payload === 'object') {
      return {
        title: '对象观察',
        sourceType: 'object',
        sourceLabel: label,
        raw: payload,
        metrics: [
          { label: '顶层键数', value: formatNumber(Object.keys(payload).length), detail: '对象顶层键数量' }
        ],
        alerts: [
          { tone: 'info', title: '对象模式', detail: '对象结构被展开成 key/value 表，适合做配置或 manifest 自检。' }
        ],
        views: [
          {
            id: 'entries',
            label: '对象键值',
            description: '适合查看配置、manifest 或统计对象。',
            rows: sanitizeRows(Object.keys(payload).map(function (key) {
              return { key: key, value: payload[key] };
            })),
            focus: '对象',
            defaultSortField: 'key'
          }
        ],
        sourceSummary: '已加载通用对象数据。'
      };
    }

    return {
      title: '原始值观察',
      sourceType: 'scalar',
      sourceLabel: label,
      raw: payload,
      metrics: [
        { label: '类型', value: typeof payload, detail: '当前数据为标量值' }
      ],
      alerts: [
        { tone: 'warn', title: '当前数据不是结构化对象', detail: '无法形成表格，只展示原始文本。' }
      ],
      views: [],
      sourceSummary: '已加载非结构化值。'
    };
  }

  async function buildDataset(path, sampleSize) {
    const text = await fetchText(path);
    const payload = parseStructuredText(text);

    if (
      payload &&
      !Array.isArray(payload) &&
      typeof payload === 'object' &&
      payload.files &&
      (Object.prototype.hasOwnProperty.call(payload, 'books_count') || Object.prototype.hasOwnProperty.call(payload, 'chapters_count'))
    ) {
      return loadManifestDataset(path, payload, sampleSize);
    }

    if (
      payload &&
      !Array.isArray(payload) &&
      typeof payload === 'object' &&
      Array.isArray(payload.shards)
    ) {
      return loadShardDataset(payload);
    }

    return loadGenericDataset(payload, path.split('/').pop() || path);
  }

  function getCurrentView() {
    if (!state.dataset || !Array.isArray(state.dataset.views)) {
      return null;
    }
    return state.dataset.views.find(function (view) {
      return view.id === state.currentViewId;
    }) || state.dataset.views[0] || null;
  }

  function buildSortCandidates(schema) {
    return schema.map(function (entry) {
      return {
        field: entry.field,
        label: entry.field + ' · ' + entry.types.join('/')
      };
    });
  }

  function renderMetrics(metrics) {
    if (!metrics.length) {
      metricGrid.innerHTML = '<div class="muted">暂无指标。</div>';
      return;
    }
    metricGrid.innerHTML = metrics.map(function (metric) {
      return [
        '<article class="inspector-metric-card">',
        '<span class="inspector-metric-label">', escapeHtml(metric.label), '</span>',
        '<strong class="inspector-metric-value">', escapeHtml(metric.value), '</strong>',
        '<span class="muted">', escapeHtml(metric.detail || ''), '</span>',
        '</article>'
      ].join('');
    }).join('');
  }

  function renderAlerts(alerts) {
    alertCount.textContent = alerts.length + ' 条';
    if (!alerts.length) {
      alertList.innerHTML = '<div class="inspector-alert tone-ok"><strong>无告警</strong><p>当前没有可提示的风险。</p></div>';
      return;
    }
    alertList.innerHTML = alerts.map(function (alert) {
      return [
        '<article class="inspector-alert tone-', escapeHtml(alert.tone || 'info'), '">',
        '<strong>', escapeHtml(alert.title), '</strong>',
        '<p>', escapeHtml(alert.detail || ''), '</p>',
        '</article>'
      ].join('');
    }).join('');
  }

  function renderSchemaTable(schema) {
    schemaSummary.textContent = schema.length + ' 个字段';
    if (!schema.length) {
      schemaEl.innerHTML = '<div class="muted">当前视图没有结构化字段。</div>';
      return;
    }
    const html = [
      '<table class="inspector-table">',
      '<thead><tr><th>字段</th><th>类型</th><th>覆盖率</th><th>去重样本</th><th>示例值</th></tr></thead>',
      '<tbody>'
    ];
    schema.forEach(function (entry) {
      html.push(
        '<tr>' +
          '<td>' + escapeHtml(entry.field) + '</td>' +
          '<td>' + escapeHtml(entry.types.join(', ')) + '</td>' +
          '<td>' + escapeHtml(String(entry.presentPct)) + '%</td>' +
          '<td>' + escapeHtml(String(entry.distinctCount)) + '</td>' +
          '<td class="inspector-cell-break">' + escapeHtml(stringifyValue(entry.example)) + '</td>' +
        '</tr>'
      );
    });
    html.push('</tbody></table>');
    schemaEl.innerHTML = html.join('');
  }

  function populateViewSelect(views) {
    viewSelect.innerHTML = views.map(function (view) {
      return '<option value="' + escapeHtml(view.id) + '">' + escapeHtml(view.label) + '</option>';
    }).join('');
    viewSelect.disabled = !views.length;
  }

  function populateSelect(select, items, defaultValue, placeholderLabel) {
    const options = ['<option value="">' + escapeHtml(placeholderLabel) + '</option>'];
    items.forEach(function (item) {
      options.push('<option value="' + escapeHtml(item.field) + '">' + escapeHtml(item.label) + '</option>');
    });
    select.innerHTML = options.join('');
    select.disabled = !items.length;
    if (defaultValue) {
      select.value = defaultValue;
    }
  }

  function compareValues(left, right) {
    const leftNumber = Number(left);
    const rightNumber = Number(right);
    const bothNumeric = Number.isFinite(leftNumber) && Number.isFinite(rightNumber) && String(left).trim() !== '' && String(right).trim() !== '';
    if (bothNumeric) {
      return leftNumber - rightNumber;
    }
    return String(left == null ? '' : left).localeCompare(String(right == null ? '' : right), 'zh-CN');
  }

  function getVisibleFields(rows, focusedField) {
    if (focusedField && focusedField !== '__all__') {
      return [focusedField];
    }
    const fields = new Set();
    rows.forEach(function (row) {
      Object.keys(row).forEach(function (key) {
        if (key !== '__row_id') {
          fields.add(key);
        }
      });
    });
    return Array.from(fields);
  }

  function renderPreview(rows, fields, meta) {
    if (!rows.length) {
      previewEl.innerHTML = '<div class="muted">没有匹配记录。</div>';
      previewSummary.textContent = '0 / 0 行';
      detailTitle.textContent = '点击表格行查看';
      detailEl.textContent = '-';
      return;
    }

    const html = [
      '<table class="inspector-table inspector-preview-table">',
      '<thead><tr><th>#</th>',
      fields.map(function (field) {
        return '<th>' + escapeHtml(field) + '</th>';
      }).join(''),
      '</tr></thead><tbody>'
    ];

    rows.forEach(function (row, index) {
      const selectedClass = index === state.selectedIndex ? ' is-selected' : '';
      html.push('<tr class="inspector-row' + selectedClass + '" data-row-index="' + index + '">');
      html.push('<td>' + escapeHtml(String(row.__row_id || index + 1)) + '</td>');
      fields.forEach(function (field) {
        html.push('<td class="inspector-cell-break">' + escapeHtml(stringifyValue(row[field])) + '</td>');
      });
      html.push('</tr>');
    });
    html.push('</tbody></table>');

    previewEl.innerHTML = html.join('');
    if (meta && meta.isTruncated) {
      previewSummary.textContent = rows.length + ' / ' + meta.matchedCount + ' 行（受展示行数限制）';
    } else if (meta) {
      previewSummary.textContent = rows.length + ' / ' + meta.matchedCount + ' 行';
    } else {
      previewSummary.textContent = rows.length + ' 行';
    }

    Array.from(previewEl.querySelectorAll('[data-row-index]')).forEach(function (rowEl) {
      rowEl.addEventListener('click', function () {
        state.selectedIndex = Number(rowEl.getAttribute('data-row-index'));
        renderCurrentView();
      });
    });

    const selectedRow = rows[state.selectedIndex] || rows[0];
    if (selectedRow) {
      detailTitle.textContent = '第 ' + selectedRow.__row_id + ' 行';
      detailEl.textContent = JSON.stringify(selectedRow, null, 2);
      if (state.selectedIndex < 0) {
        state.selectedIndex = 0;
      }
    }
  }

  function renderCurrentView() {
    const view = getCurrentView();
    if (!view) {
      renderMetrics([]);
      renderAlerts([]);
      renderSchemaTable([]);
      previewEl.innerHTML = '<div class="muted">当前数据没有可视图。</div>';
      previewSummary.textContent = '0 / 0 行';
      detailEl.textContent = '-';
      return;
    }

    const sampleSize = Math.max(5, parseInt(sampleInput.value, 10) || 20);
    const keyword = (keywordInput.value || '').trim().toLowerCase();
    const sortField = sortFieldSelect.value;
    const focusedField = columnFocusSelect.value;
    const sortDesc = sortDescInput.checked;

    const summary = summarizeRows(view.rows, sampleSize);
    const schema = summary.schema;
    const sortCandidates = buildSortCandidates(schema);
    const viewDescription = view.description || '';
    datasetBadge.textContent = state.dataset.title + ' · ' + view.label;
    sourceSummary.textContent = state.dataset.sourceSummary + ' · ' + viewDescription;

    if (sortFieldSelect.dataset.renderedView !== view.id) {
      populateSelect(sortFieldSelect, sortCandidates, view.defaultSortField || '', '默认顺序');
      sortFieldSelect.dataset.renderedView = view.id;
    }

    if (columnFocusSelect.dataset.renderedView !== view.id) {
      const focusItems = schema.map(function (entry) {
        return { field: entry.field, label: entry.field };
      });
      columnFocusSelect.innerHTML = ['<option value="__all__">全部字段</option>'].concat(focusItems.map(function (item) {
        return '<option value="' + escapeHtml(item.field) + '">' + escapeHtml(item.label) + '</option>';
      })).join('');
      columnFocusSelect.disabled = !focusItems.length;
      columnFocusSelect.dataset.renderedView = view.id;
      columnFocusSelect.value = '__all__';
    }

    let rows = view.rows.filter(function (row) {
      if (!keyword) {
        return true;
      }
      return Object.keys(row).some(function (key) {
        return key !== '__row_id' && valueForSearch(row[key]).includes(keyword);
      });
    });

    if (sortField) {
      rows = rows.slice().sort(function (left, right) {
        const diff = compareValues(left[sortField], right[sortField]);
        return sortDesc ? -diff : diff;
      });
    }

    const matchedCount = rows.length;
    const isTruncated = matchedCount > sampleSize;
    rows = rows.slice(0, sampleSize);
    state.filteredRows = rows;
    renderMetrics(state.dataset.metrics);
    renderAlerts(state.dataset.alerts);
    renderSchemaTable(schema);
    renderPreview(rows, getVisibleFields(rows, focusedField), {
      matchedCount: matchedCount,
      isTruncated: isTruncated
    });
    rawEl.textContent = JSON.stringify(state.dataset.raw, null, 2);
  }

  async function loadAndAnalyze() {
    const selectedPath = fileSelect.value === '__custom__' ? customPath.value.trim() : fileSelect.value;
    if (!selectedPath) {
      window.alert('请先选择数据文件或输入路径。');
      return;
    }

    const sampleSize = Math.max(5, parseInt(sampleInput.value, 10) || 20);
    const fetchPath = selectedPath;
    loadStatus.textContent = '加载中';
    datasetBadge.textContent = '加载中';
    sourceSummary.textContent = '正在读取 ' + selectedPath;
    metricGrid.innerHTML = '';
    alertList.innerHTML = '';
    schemaEl.innerHTML = '';
    previewEl.innerHTML = '';
    detailEl.textContent = '-';
    rawEl.textContent = '-';
    state.selectedIndex = -1;
    sortFieldSelect.dataset.renderedView = '';
    columnFocusSelect.dataset.renderedView = '';

    try {
      const dataset = await buildDataset(fetchPath, sampleSize);
      state.dataset = dataset;
      state.currentViewId = dataset.views[0] ? dataset.views[0].id : '';
      populateViewSelect(dataset.views || []);
      viewSelect.value = state.currentViewId;
      loadStatus.textContent = '已加载';
      renderCurrentView();
    } catch (error) {
      state.dataset = null;
      state.currentViewId = '';
      loadStatus.textContent = '加载失败';
      datasetBadge.textContent = '失败';
      sourceSummary.textContent = '读取失败: ' + error.message;
      renderMetrics([]);
      renderAlerts([{ tone: 'danger', title: '加载失败', detail: error.message }]);
      renderSchemaTable([]);
      previewEl.innerHTML = '<div class="muted">请检查静态服务器路径或文件格式。</div>';
      previewSummary.textContent = '0 / 0 行';
      detailEl.textContent = '-';
      rawEl.textContent = error.stack || error.message;
    }
  }

  function downloadText(content, fileName, type) {
    const blob = new Blob([content], { type: type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function toggleCustomPath() {
    const isCustom = fileSelect.value === '__custom__';
    customPathWrap.hidden = !isCustom;
    if (isCustom) {
      customPath.focus();
    }
  }

  function enableDragScroll(container) {
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startScrollLeft = 0;
    let startScrollTop = 0;

    container.addEventListener('mousedown', function (event) {
      if (event.button !== 0) {
        return;
      }
      if (event.target.closest('button, a, input, select, textarea')) {
        return;
      }
      isDragging = true;
      startX = event.clientX;
      startY = event.clientY;
      startScrollLeft = container.scrollLeft;
      startScrollTop = container.scrollTop;
      container.classList.add('is-dragging');
    });

    window.addEventListener('mousemove', function (event) {
      if (!isDragging) {
        return;
      }
      const deltaX = event.clientX - startX;
      const deltaY = event.clientY - startY;
      container.scrollLeft = startScrollLeft - deltaX;
      container.scrollTop = startScrollTop - deltaY;
    });

    window.addEventListener('mouseup', function () {
      if (!isDragging) {
        return;
      }
      isDragging = false;
      container.classList.remove('is-dragging');
    });

    container.addEventListener('mouseleave', function () {
      if (!isDragging) {
        return;
      }
      isDragging = false;
      container.classList.remove('is-dragging');
    });
  }

  fileSelect.addEventListener('change', function () {
    toggleCustomPath();
  });

  viewSelect.addEventListener('change', function () {
    state.currentViewId = viewSelect.value;
    state.selectedIndex = -1;
    sortFieldSelect.dataset.renderedView = '';
    columnFocusSelect.dataset.renderedView = '';
    renderCurrentView();
  });

  [keywordInput, sampleInput, sortFieldSelect, columnFocusSelect, sortDescInput].forEach(function (element) {
    element.addEventListener('input', renderCurrentView);
    element.addEventListener('change', renderCurrentView);
  });

  customPath.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
      loadAndAnalyze();
    }
  });

  loadBtn.addEventListener('click', loadAndAnalyze);

  tableWraps.forEach(function (wrap) {
    enableDragScroll(wrap);
  });

  exportBtn.addEventListener('click', function () {
    const view = getCurrentView();
    if (!view || !state.filteredRows.length) {
      window.alert('当前没有可导出的表格数据。');
      return;
    }
    const fields = getVisibleFields(state.filteredRows, columnFocusSelect.value);
    const csv = toCsv(state.filteredRows, fields);
    downloadText(csv, (view.id || 'preview') + '.csv', 'text/csv;charset=utf-8');
  });

  downloadJsonBtn.addEventListener('click', function () {
    if (!state.dataset) {
      window.alert('当前没有可下载的数据。');
      return;
    }
    downloadText(JSON.stringify(state.dataset.raw, null, 2), (state.dataset.sourceType || 'dataset') + '.json', 'application/json;charset=utf-8');
  });

  toggleCustomPath();
  loadAndAnalyze();
})();
