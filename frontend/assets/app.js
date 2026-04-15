(function () {
    const treeContainer = document.getElementById('doc-tree');
    const searchInput = document.getElementById('doc-search');
    const featuredContainer = document.getElementById('featured-posts');
    const allPostsContainer = document.getElementById('all-posts');
    const tagFilter = document.getElementById('tag-filter');
    const feedCount = document.getElementById('feed-count');
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');
    const overlay = document.getElementById('sidebar-overlay');
    const themeToggle = document.getElementById('theme-toggle');
    const randomPostBtn = document.getElementById('random-post');
    const copyCmdBtn = document.getElementById('copy-cmd');
    const copyTip = document.getElementById('copy-tip');
    const toTopBtn = document.getElementById('to-top');
    const publishCmd = document.getElementById('publish-cmd');
    const readerPanel = document.getElementById('reader-panel');
    const readerTitle = document.getElementById('reader-title');
    const readerPath = document.getElementById('reader-path');
    const readerContent = document.getElementById('reader-content');
    const readerBack = document.getElementById('reader-back');

    let allFiles = [];
    let currentTag = '全部';

    function createDocHref(path) {
        return '?doc=' + encodeURIComponent(path);
    }

    function toggleSidebar(force) {
        const shouldOpen = typeof force === 'boolean' ? force : !sidebar.classList.contains('open');
        sidebar.classList.toggle('open', shouldOpen);
        if (overlay) {
            overlay.classList.toggle('show', shouldOpen);
            overlay.setAttribute('aria-hidden', shouldOpen ? 'false' : 'true');
        }
    }

    toggleBtn.addEventListener('click', function () {
        toggleSidebar();
    });

    if (overlay) {
        overlay.addEventListener('click', function () {
            toggleSidebar(false);
        });
    }

    document.addEventListener('click', function (event) {
        if (!sidebar.classList.contains('open')) return;
        const insideSidebar = sidebar.contains(event.target);
        const isToggleBtn = toggleBtn.contains(event.target);
        if (!insideSidebar && !isToggleBtn) {
            toggleSidebar(false);
        }
    });

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
            toggleSidebar(false);
        }
    });

    function setupTheme() {
        const storageKey = 'blog-theme';
        const saved = localStorage.getItem(storageKey);
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        const theme = saved || (prefersDark ? 'dark' : 'light');

        document.body.setAttribute('data-theme', theme);
        if (themeToggle) {
            themeToggle.textContent = theme === 'dark' ? '日间' : '夜间';
            themeToggle.addEventListener('click', function () {
                const next = document.body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                document.body.setAttribute('data-theme', next);
                localStorage.setItem(storageKey, next);
                themeToggle.textContent = next === 'dark' ? '日间' : '夜间';
            });
        }
    }

    function collectFiles(node, bucket, parentPath) {
        if (!node) return;

        if (node.type === 'file') {
            const rawPath = node.path || '';
            const cleaned = rawPath.replace(/^docs\//, '');
            const parts = cleaned.split('/').filter(Boolean);
            const category = parts.length > 1 ? parts[0] : '未分类';
            bucket.push({
                name: node.name,
                path: rawPath,
                category: category,
                parentPath: parentPath || 'docs'
            });
            return;
        }

        if (node.type === 'folder' && Array.isArray(node.children)) {
            node.children.forEach(function (child) {
                collectFiles(child, bucket, (parentPath ? parentPath + '/' : '') + node.name);
            });
        }
    }

    function createTreeNode(node) {
        if (node.type === 'folder') {
            const details = document.createElement('details');
            const summary = document.createElement('summary');
            summary.textContent = node.name;
            details.appendChild(summary);

            if (Array.isArray(node.children) && node.children.length) {
                const list = document.createElement('ul');
                node.children.forEach(function (child) {
                    list.appendChild(createTreeNode(child));
                });
                details.appendChild(list);
            }

            return details;
        }

        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = createDocHref(node.path);
        link.textContent = node.name;
        item.appendChild(link);
        return item;
    }

    function updateStats(files, tree) {
        const folders = countFolders(tree);
        document.getElementById('stat-files').textContent = String(files.length);
        document.getElementById('stat-folders').textContent = String(folders);
        document.getElementById('stat-updated').textContent = new Date().toLocaleDateString('zh-CN');
    }

    function countFolders(node) {
        if (!node || node.type !== 'folder') return 0;
        let total = 1;
        if (Array.isArray(node.children)) {
            node.children.forEach(function (child) {
                total += countFolders(child);
            });
        }
        return total;
    }

    function renderFeatured(files) {
        featuredContainer.innerHTML = '';
        const top = files.slice(0, 6);

        top.forEach(function (file) {
            const card = document.createElement('article');
            card.className = 'post-card';
            card.innerHTML =
                '<h3 class="post-title"><a href="' + createDocHref(file.path) + '">' + file.name + '</a></h3>' +
                '<p class="post-meta">分类：' + file.category + '</p>' +
                '<p class="post-meta">路径：' + file.path + '</p>';
            featuredContainer.appendChild(card);
        });
    }

    function renderTags(files) {
        const categories = Array.from(new Set(files.map(function (f) {
            return f.category;
        }))).sort();
        const tags = ['全部'].concat(categories);
        tagFilter.innerHTML = '';

        tags.forEach(function (tag) {
            const btn = document.createElement('button');
            btn.className = 'tag-btn' + (tag === currentTag ? ' active' : '');
            btn.textContent = tag;
            btn.addEventListener('click', function () {
                currentTag = tag;
                renderTags(allFiles);
                renderAllPosts(filterFiles());
            });
            tagFilter.appendChild(btn);
        });
    }

    function filterFiles() {
        const keyword = (searchInput.value || '').trim().toLowerCase();

        return allFiles.filter(function (file) {
            const matchTag = currentTag === '全部' || file.category === currentTag;
            const matchText = !keyword ||
                file.name.toLowerCase().includes(keyword) ||
                file.path.toLowerCase().includes(keyword);
            return matchTag && matchText;
        });
    }

    function renderAllPosts(files) {
        allPostsContainer.innerHTML = '';
        feedCount.textContent = files.length + ' 篇';

        if (!files.length) {
            allPostsContainer.innerHTML = '<p class="muted">没有匹配的文章。</p>';
            return;
        }

        files.forEach(function (file) {
            const row = document.createElement('article');
            row.className = 'post-row';
            row.innerHTML =
                '<h3 class="post-title"><a href="' + createDocHref(file.path) + '">' + file.name + '</a></h3>' +
                '<p class="post-meta">' + file.category + ' · ' + file.path + '</p>';
            allPostsContainer.appendChild(row);
        });
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function resolveAssetPath(assetPath, docPath) {
        if (!docPath || assetPath.startsWith('http://') || assetPath.startsWith('https://') || assetPath.startsWith('/')) {
            return assetPath;
        }
        var docDir = docPath.substring(0, docPath.lastIndexOf('/'));
        return docDir + '/' + assetPath;
    }

    function renderInline(text, docPath) {
        var result = text;
        var images = [];
        var links = [];

        // Extract images first
        result = result.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function (match, alt, src) {
            var resolvedSrc = resolveAssetPath(src, docPath);
            images.push('<img src="' + resolvedSrc + '" alt="' + alt + '" />');
            return '\x00IMG' + (images.length - 1) + '\x00';
        });

        // Extract links
        result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (match, text, url) {
            links.push('<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + text + '</a>');
            return '\x00LINK' + (links.length - 1) + '\x00';
        });

        // Escape HTML
        var escaped = escapeHtml(result);

        // Restore images
        escaped = escaped.replace(/\x00IMG(\d+)\x00/g, function (match, index) {
            return images[parseInt(index)];
        });

        // Restore links
        escaped = escaped.replace(/\x00LINK(\d+)\x00/g, function (match, index) {
            return links[parseInt(index)];
        });

        // Handle inline formatting
        return escaped
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>');
    }

    function renderInlineWithoutPath(text) {
        return renderInline(text, null);
    }

    function renderMarkdown(markdown, docPath) {
        const lines = markdown.replace(/\r\n/g, '\n').split('\n');
        const html = [];
        let inCode = false;
        let codeBuffer = [];
        let listType = null;
        let tableRows = [];
        let inMath = false;
        let mathBuffer = [];

        function closeList() {
            if (listType) {
                html.push(listType === 'ol' ? '</ol>' : '</ul>');
                listType = null;
            }
        }

        function closeTable() {
            if (!tableRows.length) { return; }
            var rows = tableRows;
            tableRows = [];
            if (rows.length < 2 || !/^[\s|:\-]+$/.test(rows[1])) {
                rows.forEach(function (r) { html.push('<p>' + renderInline(r, docPath) + '</p>'); });
                return;
            }
            function parseCells(row) {
                var parts = row.split('|');
                if (parts[0].trim() === '') { parts = parts.slice(1); }
                if (parts.length && parts[parts.length - 1].trim() === '') { parts = parts.slice(0, -1); }
                return parts.map(function (c) { return c.trim(); });
            }
            var headers = parseCells(rows[0]);
            var thead = '<thead><tr>' + headers.map(function (h) {
                return '<th>' + renderInline(h, docPath) + '</th>';
            }).join('') + '</tr></thead>';
            var tbody = rows.length > 2 ? '<tbody>' + rows.slice(2).map(function (row) {
                return '<tr>' + parseCells(row).map(function (c) {
                    return '<td>' + renderInline(c, docPath) + '</td>';
                }).join('') + '</tr>';
            }).join('') + '</tbody>' : '';
            html.push('<div class="table-wrap"><table>' + thead + tbody + '</table></div>');
        }

        function closeBlocks() {
            closeList();
            closeTable();
        }

        for (let i = 0; i < lines.length; i += 1) {
            const line = lines[i];

            // Block math: $$ ... $$ (multi-line)
            if (!inCode && line.trim() === '$$') {
                closeBlocks();
                if (!inMath) {
                    inMath = true;
                    mathBuffer = [];
                } else {
                    html.push('<div class="math-display">$$' + mathBuffer.join('\n') + '$$</div>');
                    inMath = false;
                }
                continue;
            }

            if (inMath) {
                mathBuffer.push(line);
                continue;
            }

            if (line.trim().startsWith('```')) {
                closeBlocks();
                if (!inCode) {
                    inCode = true;
                    codeBuffer = [];
                } else {
                    html.push('<pre><code>' + escapeHtml(codeBuffer.join('\n')) + '</code></pre>');
                    inCode = false;
                }
                continue;
            }

            if (inCode) {
                codeBuffer.push(line);
                continue;
            }

            if (!line.trim()) {
                closeBlocks();
                continue;
            }

            const heading = line.match(/^(#{1,4})\s+(.*)$/);
            if (heading) {
                closeBlocks();
                const level = heading[1].length;
                html.push('<h' + level + '>' + renderInline(heading[2], docPath) + '</h' + level + '>');
                continue;
            }

            // GFM table row
            if (line.trim().startsWith('|')) {
                closeList();
                tableRows.push(line.trim());
                continue;
            }

            const ordered = line.match(/^\d+\.\s+(.*)$/);
            if (ordered) {
                closeTable();
                if (listType !== 'ol') {
                    closeList();
                    listType = 'ol';
                    html.push('<ol>');
                }
                html.push('<li>' + renderInline(ordered[1], docPath) + '</li>');
                continue;
            }

            const unordered = line.match(/^[-*+]\s+(.*)$/);
            if (unordered) {
                closeTable();
                if (listType !== 'ul') {
                    closeList();
                    listType = 'ul';
                    html.push('<ul>');
                }
                html.push('<li>' + renderInline(unordered[1], docPath) + '</li>');
                continue;
            }

            if (line.startsWith('> ')) {
                closeBlocks();
                html.push('<blockquote><p>' + renderInline(line.slice(2), docPath) + '</p></blockquote>');
                continue;
            }

            closeBlocks();
            html.push('<p>' + renderInline(line, docPath) + '</p>');
        }

        closeBlocks();
        return html.join('');
    }

    async function renderReaderFromLocation() {
        const params = new URLSearchParams(window.location.search);
        const docPath = params.get('doc');

        if (!docPath) {
            if (readerPanel) {
                readerPanel.hidden = true;
            }
            return;
        }

        if (!readerPanel || !readerTitle || !readerContent || !readerPath) {
            return;
        }

        readerPanel.hidden = false;
        readerTitle.textContent = '加载中...';
        readerPath.textContent = docPath;
        readerContent.innerHTML = '<p class="muted">正在加载文档...</p>';

        try {
            // 当 frontend 在子目录时，文档资源位于上一级的 docs/ 下
            let docFetchPath = docPath;
            if (docPath && !docPath.startsWith('http://') && !docPath.startsWith('https://') && !docPath.startsWith('/')) {
                docFetchPath = '../' + docPath;
            }

            const response = await fetch(docFetchPath, {cache: 'no-store'});
            if (!response.ok) {
                throw new Error('无法加载文档');
            }
            const markdown = await response.text();
            const title = markdown.match(/^#\s+(.+)$/m);
            readerTitle.textContent = title ? title[1] : docPath.split('/').pop();
            readerContent.innerHTML = renderMarkdown(markdown, docFetchPath);
            if (typeof window.renderMathInElement === 'function') {
                window.renderMathInElement(readerContent, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false}
                    ],
                    throwOnError: false
                });
            }
            requestAnimationFrame(function () {
                readerPanel.scrollIntoView({behavior: 'smooth', block: 'start'});
            });
        } catch (err) {
            readerTitle.textContent = '加载失败';
            readerContent.innerHTML = '<p class="muted">文档加载失败：' + err.message + '</p>';
        }
    }

    function applyTreeSearch(term) {
        const keyword = term.trim().toLowerCase();
        const detailsList = treeContainer.querySelectorAll('details');
        const items = treeContainer.querySelectorAll('li');

        if (!keyword) {
            detailsList.forEach(function (d) {
                d.style.display = '';
                d.open = false;
            });
            items.forEach(function (li) {
                li.style.display = '';
            });
            return;
        }

        items.forEach(function (li) {
            const show = li.textContent.toLowerCase().includes(keyword);
            li.style.display = show ? '' : 'none';
        });

        detailsList.forEach(function (d) {
            const show = d.textContent.toLowerCase().includes(keyword);
            d.style.display = show ? '' : 'none';
            d.open = show;
        });
    }

    searchInput.addEventListener('input', function () {
        const filtered = filterFiles();
        renderAllPosts(filtered);
        renderFeatured(filtered.length ? filtered : allFiles);
        applyTreeSearch(searchInput.value || '');
    });

    function setupRandomPost() {
        if (!randomPostBtn) return;

        randomPostBtn.addEventListener('click', function () {
            const source = filterFiles();
            const pool = source.length ? source : allFiles;
            if (!pool.length) return;
            const pick = pool[Math.floor(Math.random() * pool.length)];
            window.location.href = createDocHref(pick.path);
        });
    }

    function setupCopyCommand() {
        if (!copyCmdBtn || !publishCmd) return;

        copyCmdBtn.addEventListener('click', async function () {
            const text = publishCmd.innerText.trim();
            try {
                await navigator.clipboard.writeText(text);
                if (copyTip) copyTip.textContent = '已复制到剪贴板';
            } catch (err) {
                if (copyTip) copyTip.textContent = '复制失败，请手动复制';
            }
            setTimeout(function () {
                if (copyTip) copyTip.textContent = '可直接粘贴到终端执行';
            }, 1800);
        });
    }

    function setupToTop() {
        if (!toTopBtn) return;

        window.addEventListener('scroll', function () {
            const show = window.scrollY > 420;
            toTopBtn.classList.toggle('show', show);
        });

        toTopBtn.addEventListener('click', function () {
            window.scrollTo({top: 0, behavior: 'smooth'});
        });
    }

    function setupReaderBack() {
        if (!readerBack) return;
        readerBack.addEventListener('click', function () {
            window.history.pushState({}, '', './');
            renderReaderFromLocation();
            window.scrollTo({top: 0, behavior: 'smooth'});
        });
    }

    async function init() {
        setupTheme();
        setupRandomPost();
        setupCopyCommand();
        setupToTop();
        setupReaderBack();

        window.addEventListener('popstate', function () {
            renderReaderFromLocation();
        });

        try {
            const response = await fetch('../docs/index.json', {cache: 'no-store'});
            if (!response.ok) {
                throw new Error('无法加载 docs/index.json');
            }

            const tree = await response.json();
            treeContainer.innerHTML = '';
            treeContainer.appendChild(createTreeNode(tree));

            const files = [];
            collectFiles(tree, files, '');
            allFiles = files.filter(function (f) {
                return f.path.toLowerCase().endsWith('.md');
            });

            updateStats(allFiles, tree);
            renderTags(allFiles);
            renderFeatured(allFiles);
            renderAllPosts(allFiles);
            renderReaderFromLocation();
        } catch (err) {
            treeContainer.innerHTML = '<p class="muted">目录加载失败：' + err.message + '</p>';
            featuredContainer.innerHTML = '<p class="muted">暂时无法加载文章列表。</p>';
            allPostsContainer.innerHTML = '<p class="muted">暂时无法加载文章列表。</p>';
        }
    }

    init();
})();
