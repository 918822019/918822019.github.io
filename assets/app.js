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

  let allFiles = [];
  let currentTag = '全部';

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
    link.href = node.path;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
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
        '<h3 class="post-title"><a href="' + file.path + '" target="_blank" rel="noopener noreferrer">' + file.name + '</a></h3>' +
        '<p class="post-meta">分类：' + file.category + '</p>' +
        '<p class="post-meta">路径：' + file.path + '</p>';
      featuredContainer.appendChild(card);
    });
  }

  function renderTags(files) {
    const categories = Array.from(new Set(files.map(function (f) { return f.category; }))).sort();
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
        '<h3 class="post-title"><a href="' + file.path + '" target="_blank" rel="noopener noreferrer">' + file.name + '</a></h3>' +
        '<p class="post-meta">' + file.category + ' · ' + file.path + '</p>';
      allPostsContainer.appendChild(row);
    });
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
      window.open(pick.path, '_blank', 'noopener,noreferrer');
    });
  }

  async function init() {
    setupTheme();
    setupRandomPost();

    try {
      const response = await fetch('docs/index.json', { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('无法加载 docs/index.json');
      }

      const tree = await response.json();
      treeContainer.innerHTML = '';
      treeContainer.appendChild(createTreeNode(tree));

      const files = [];
      collectFiles(tree, files, '');
      allFiles = files.filter(function (f) { return f.path.toLowerCase().endsWith('.md'); });

      updateStats(allFiles, tree);
      renderTags(allFiles);
      renderFeatured(allFiles);
      renderAllPosts(allFiles);
    } catch (err) {
      treeContainer.innerHTML = '<p class="muted">目录加载失败：' + err.message + '</p>';
      featuredContainer.innerHTML = '<p class="muted">暂时无法加载文章列表。</p>';
      allPostsContainer.innerHTML = '<p class="muted">暂时无法加载文章列表。</p>';
    }
  }

  init();
})();
