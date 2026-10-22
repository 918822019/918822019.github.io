(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const categoryNames = {
    infra: "推理与系统",
    模型相关的论文: "模型与论文",
    架构设计: "架构设计",
    数学: "数学基础",
    AI搜索: "AI 搜索",
    RAG: "RAG 与检索",
    其他: "其他笔记",
  };
  const categorySymbols = {
    infra: "⌘",
    模型相关的论文: "◇",
    架构设计: "▦",
    数学: "∑",
    AI搜索: "↗",
    RAG: "◎",
  };
  const pcodePath = "docs/infra/推理优化/pcode-report.md";
  let articles = [],
    activeCategory = "全部",
    readRequest = 0,
    headingObserver;
  const textCache = new Map();
  const escape = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const href = (path) => "/?doc=" + encodeURIComponent(path);
  const categoryLabel = (key) => categoryNames[key] || key;
  const shortDate = (date) => (date ? date.replaceAll("-", ".") : "");
  const docPath = () => new URLSearchParams(location.search).get("doc");
  const notice = (message) =>
    '<div class="empty-state"><strong>' +
    escape(message) +
    "</strong><span>你可以返回所有手记，或稍后重试。</span></div>";

  function collect(node) {
    if (node.type === "file" && node.path.endsWith(".md")) {
      const parts = node.path.split("/");
      articles.push({
        ...node,
        category: parts.length > 2 ? parts[1] : "其他",
        title: node.title || node.name.replace(/\.md$/, ""),
        excerpt: node.excerpt || "打开这篇手记，阅读完整的思考与实践记录。",
        readingMinutes: node.readingMinutes || 1,
      });
    }
    (node.children || []).forEach(collect);
  }
  function publicArticles() {
    return articles.filter((a) => !/\/README\.md$/i.test(a.path));
  }
  function sortArticles(list) {
    return [...list].sort((a, b) =>
      $("sort-order").value === "title"
        ? a.title.localeCompare(b.title, "zh-CN")
        : (b.updated || "").localeCompare(a.updated || "") ||
          a.title.localeCompare(b.title, "zh-CN"),
    );
  }
  function articleCard(a) {
    return `<article class="post-card"><span class="post-category">${escape(categoryLabel(a.category))}</span><h3><a href="${href(a.path)}">${escape(a.title)}</a></h3><p class="post-excerpt">${escape(a.excerpt)}</p><div class="post-bottom">${a.updated ? `<time datetime="${escape(a.updated)}">${shortDate(a.updated)}</time><span>·</span>` : ""}<span>约 ${a.readingMinutes} 分钟</span><span class="post-arrow" aria-hidden="true">↗</span></div></article>`;
  }
  function renderPosts() {
    const filtered = sortArticles(
      publicArticles().filter(
        (a) => activeCategory === "全部" || a.category === activeCategory,
      ),
    );
    $("post-list").innerHTML = filtered.length
      ? filtered.map(articleCard).join("")
      : notice("这个专题暂时还没有文章");
    $("result-count").textContent =
      `${String(filtered.length).padStart(2, "0")} 篇文章`;
    $("category-filters")
      .querySelectorAll("button")
      .forEach((b) => {
        const active = b.dataset.category === activeCategory;
        b.classList.toggle("active", active);
        b.setAttribute("aria-pressed", String(active));
      });
  }
  function chooseCategory(category) {
    activeCategory = category;
    renderPosts();
    $("notes").scrollIntoView({ behavior: "smooth" });
  }
  function renderHome() {
    const files = publicArticles();
    const categories = [...new Set(files.map((a) => a.category))];
    $("article-count").textContent = String(files.length).padStart(2, "0");
    $("topic-count").textContent = String(categories.length).padStart(2, "0");
    $("category-filters").innerHTML = ["全部", ...categories]
      .map(
        (c) =>
          `<button class="filter-button" data-category="${escape(c)}" aria-pressed="false">${escape(c === "全部" ? c : categoryLabel(c))}</button>`,
      )
      .join("");
    $("category-filters")
      .querySelectorAll("button")
      .forEach((b) =>
        b.addEventListener("click", () => {
          activeCategory = b.dataset.category;
          renderPosts();
        }),
      );
    $("topic-list").innerHTML = categories
      .map(
        (c) =>
          `<button class="topic-button" data-category="${escape(c)}"><span class="topic-symbol" aria-hidden="true">${categorySymbols[c] || "◇"}</span><span>${escape(categoryLabel(c))}</span><span class="topic-number">${String(files.filter((a) => a.category === c).length).padStart(2, "0")}</span><span class="topic-arrow" aria-hidden="true">↗</span></button>`,
      )
      .join("");
    $("topic-list")
      .querySelectorAll("button")
      .forEach((b) =>
        b.addEventListener("click", () => chooseCategory(b.dataset.category)),
      );
    const featured =
      files.find((a) => a.path === pcodePath) || sortArticles(files)[0];
    if (featured) {
      $("spotlight").classList.remove("loading-placeholder");
      const visual =
        featured.path === pcodePath
          ? `<div class="spotlight-visual" aria-label="Pcode 双分类头示意与实验指标"><div class="visual-label"><span>PCODE / PARALLEL READOUT</span><span>896 → 390 × 2</span></div><div class="pcode-flow"><div class="flow-node">Hidden<br>state</div><span class="flow-arrow" aria-hidden="true">→</span><div class="flow-heads"><span>Head A</span><span>Head B</span></div><span class="flow-arrow" aria-hidden="true">→</span><div class="flow-result">Token</div></div><div class="visual-metrics"><div><strong>29.7<span class="metric-unit"> MB</span></strong><small>E3 draft 侧交付</small></div><div><strong>2.70<span class="metric-unit"> ×</span></strong><small>统一 GPU 对照 / 相对 AR</small></div></div></div>`
          : `<div class="spotlight-visual"><p class="eyebrow">A NOTE WORTH READING</p><span class="about-mark">n·</span><p>保持好奇，持续构建。</p></div>`;
      $("spotlight").innerHTML =
        `<div class="spotlight-copy"><span class="category-pill">精选手记 · ${escape(categoryLabel(featured.category))}</span><h2><a href="${href(featured.path)}">${escape(featured.title)}</a></h2><p>${escape(featured.path === pcodePath ? "用并行双码头压缩草稿模型的读出成本。从设计与训练，到 GPU 对照和手机验收，梳理紧凑交付的价值与边界。" : featured.excerpt)}</p><a class="text-link" href="${href(featured.path)}">阅读这篇报告 <span aria-hidden="true">↗</span></a></div>${visual}`;
    } else {
      $("spotlight").innerHTML = notice("还没有可展示的文章");
    }
    renderPosts();
  }

  let searchRequest = 0;
  async function renderSearch() {
    const request = ++searchRequest;
    const query = $("search-input").value.trim();
    $("search-count").textContent = "正在检索文章正文…";
    const { matches, fullText } = await Notebook.search(query);
    if (request !== searchRequest) return;
    $("search-count").textContent =
      (fullText ? "全文搜索" : "正文索引暂不可用，仅搜索标题与摘要") +
      (query ? ` · 找到 ${matches.length} 篇手记` : " · 输入关键词开始检索");
    $("search-results").innerHTML = matches.length
      ? matches
          .map(
            ({ article: a, snippet }) =>
              `<a class="search-result" href="${href(a.path)}"><strong>${NotebookCore.highlight(a.title, query)}</strong><span>${escape(a.category)} · 约 ${a.readingMinutes} 分钟</span><p class="search-snippet">${NotebookCore.highlight(snippet, query)}</p></a>`,
          )
          .join("")
      : '<div class="empty-state"><strong>还没有找到这条线索</strong><span>试试「Pcode」「量化」「TVM」等关键词；多个词以空格分隔。</span></div>';
  }
  function openSearch() {
    renderSearch();
    $("search-dialog").showModal();
    $("search-input").focus();
  }
  function updateTheme() {
    const dark = document.documentElement.dataset.theme === "dark";
    $("theme-toggle").setAttribute(
      "aria-label",
      dark ? "切换到浅色模式" : "切换到深色模式",
    );
    $("theme-icon").textContent = dark ? "☼" : "◐";
    document.querySelector('meta[name="theme-color"]').content = dark
      ? "#181e1b"
      : "#f7f7f2";
  }
  async function copyText(button, text, label) {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "已复制 ✓";
    } catch {
      button.textContent = "复制失败，请手动选择";
    }
    setTimeout(() => {
      button.textContent = label;
    }, 1800);
  }

  // Keep TeX out of Markdown's escaping/formatting pass, then render it locally.
  function renderMarkdown(markdown) {
    const math = [];
    const protectedMarkdown = markdown.replace(
      /(```[\s\S]*?```|`[^`\n]+`)|(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|(?<!\\)\$(?!\s)[^$\n]+?(?<!\s)\$)/g,
      (match, code, formula) => {
        if (code) return code;
        math.push(formula);
        return `PCODEMATHTOKEN${math.length - 1}ENDTOKEN`;
      },
    );
    let html = DOMPurify.sanitize(
      marked.parse(protectedMarkdown, { gfm: true }),
      { USE_PROFILES: { html: true } },
    );
    html = html.replace(/PCODEMATHTOKEN(\d+)ENDTOKEN/g, (_, i) =>
      escape(math[Number(i)]),
    );
    return html;
  }
  function decorateArticle(path) {
    const content = $("reader-content");
    const first = content.firstElementChild;
    if (first?.tagName === "H1") first.remove();
    content.querySelectorAll("h1").forEach((h) => {
      const replacement = document.createElement("h2");
      replacement.innerHTML = h.innerHTML;
      h.replaceWith(replacement);
    });
    content.querySelectorAll("img").forEach((img) => {
      const source = img.getAttribute("src");
      if (source)
        img.src = new URL(source, new URL("/" + path, location.origin)).href;
      img.loading = "lazy";
      img.decoding = "async";
    });
    content.querySelectorAll("a[href]").forEach((a) => {
      const raw = a.getAttribute("href");
      if (raw.startsWith("#")) return;
      const url = new URL(raw, new URL("/" + path, location.origin));
      if (url.origin === location.origin && url.pathname.endsWith(".md")) {
        a.href = href(decodeURIComponent(url.pathname.slice(1))) + url.hash;
      } else {
        a.href = url.href;
        if (url.origin !== location.origin) {
          a.target = "_blank";
          a.rel = "noopener noreferrer";
        }
      }
    });
    content.querySelectorAll("table").forEach((table) => {
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      wrap.tabIndex = 0;
      wrap.setAttribute("role", "region");
      wrap.setAttribute("aria-label", "数据表格，可横向滚动");
      table.before(wrap);
      wrap.append(table);
    });
    content.querySelectorAll("pre").forEach((pre) => {
      const wrapper = document.createElement("div");
      wrapper.className = "code-block";
      const toolbar = document.createElement("div");
      toolbar.className = "code-toolbar";
      const lang = document.createElement("span");
      lang.textContent =
        pre
          .querySelector("code")
          ?.className.replace("language-", "")
          .toUpperCase() || "CODE";
      const copy = document.createElement("button");
      copy.textContent = "复制代码";
      copy.addEventListener("click", () =>
        copyText(copy, pre.textContent, "复制代码"),
      );
      pre.before(wrapper);
      wrapper.append(toolbar, pre);
      toolbar.append(lang, copy);
      const code = pre.querySelector("code");
      if (code && window.Prism) Prism.highlightElement(code);
      if (pre.textContent.split("\n").length > 16) {
        wrapper.classList.add("code-collapsed");
        const fold = document.createElement("button");
        fold.className = "code-fold";
        fold.textContent = "展开完整代码";
        fold.setAttribute("aria-expanded", "false");
        fold.addEventListener("click", () => {
          const folded = wrapper.classList.toggle("code-collapsed");
          fold.setAttribute("aria-expanded", String(!folded));
          fold.textContent = folded ? "展开完整代码" : "收起代码";
        });
        wrapper.append(fold);
      }
    });
    window.Experiments?.enhance(path, content);
    renderMathInElement(content, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      trust: false,
    });
    const headings = [...content.querySelectorAll("h2,h3")];
    const ids = new Set();
    headings.forEach((h, i) => {
      let id =
        h.textContent
          .trim()
          .toLowerCase()
          .replace(/\s+/g, "-")
          .replace(/[^\p{L}\p{N}_-]/gu, "") || `section-${i + 1}`;
      if (ids.has(id)) id += "-" + (i + 1);
      ids.add(id);
      h.id = id;
    });
    $("article-toc").innerHTML = headings.length
      ? headings
          .map(
            (h) =>
              `<a class="toc-link${h.tagName === "H3" ? " sub" : ""}" href="#${encodeURIComponent(h.id)}">${escape(h.textContent)}</a>`,
          )
          .join("")
      : '<span class="toc-link">这是一篇短手记</span>';
    headingObserver?.disconnect();
    headingObserver = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort(
            (a, b) => a.boundingClientRect.top - b.boundingClientRect.top,
          )[0];
        if (visible)
          $("article-toc")
            .querySelectorAll("a")
            .forEach((a) => {
              const active =
                decodeURIComponent(a.hash.slice(1)) === visible.target.id;
              a.classList.toggle("active", active);
              if (active) a.setAttribute("aria-current", "location");
              else a.removeAttribute("aria-current");
            });
      },
      { rootMargin: "-100px 0px -60% 0px" },
    );
    headings.forEach((h) => headingObserver.observe(h));
  }

  async function renderRoute() {
    const path = docPath();
    const request = ++readRequest;
    Notebook.leaveArticle();
    const params = new URLSearchParams(location.search);
    const view = ["library", "archive", "series"].includes(params.get("view"))
      ? params.get("view")
      : null;
    $("collection-view").hidden = !!path || !view;
    $("home-view").hidden = Boolean(path) || !!view;
    $("reader-view").hidden = !path;
    $("reading-progress").style.width = "0";
    document
      .querySelectorAll(".main-nav a")
      .forEach((a) =>
        a.classList.toggle(
          "active",
          !path &&
            (a.hash === location.hash ||
              (!location.hash && a.dataset.nav === "notes")),
        ),
      );
    if (!path && view) {
      headingObserver?.disconnect();
      document
        .querySelectorAll(".main-nav > a")
        .forEach((a) => a.classList.toggle("active", a.dataset.nav === view));
      await Notebook.renderCollection(view, params);
      return;
    }
    if (!path) {
      document.title = "918822019 · 技术手记";
      headingObserver?.disconnect();
      return;
    }
    const article = articles.find((a) => a.path === path);
    $("reader-title").textContent = article?.title || "正在加载…";
    $("reader-category").textContent = article
      ? categoryLabel(article.category)
      : "技术手记";
    $("reader-date").textContent = article?.updated
      ? `更新于 ${shortDate(article.updated)}`
      : "";
    $("reader-time").textContent = article
      ? `约 ${article.readingMinutes} 分钟阅读`
      : "";
    $("reader-content").innerHTML =
      '<p class="reader-loading">正在翻开这篇手记…</p>';
    $("article-toc").innerHTML = "";
    $("related-posts").innerHTML = "";
    try {
      if (!article) throw new Error("文章不存在或尚未收录");
      let markdown = textCache.get(path);
      if (!markdown) {
        const response = await fetch("/" + path);
        if (!response.ok) throw new Error("文章加载失败");
        markdown = await response.text();
        textCache.set(path, markdown);
      }
      if (request !== readRequest) return;
      document.title = article.title + " · 918822019";
      $("reader-content").innerHTML = renderMarkdown(markdown);
      decorateArticle(path);
      Notebook.setupArticle(article);
      const others = sortArticles(
        publicArticles().filter((a) => a.path !== path),
      );
      const related = [
        ...others.filter((a) => a.category === article.category),
        ...others.filter((a) => a.category !== article.category),
      ].slice(0, 2);
      $("related-posts").innerHTML = related.map(articleCard).join("");
      if (location.hash) requestAnimationFrame(scrollToHash);
      updateProgress();
    } catch (error) {
      if (request !== readRequest) return;
      $("reader-title").textContent = "这篇手记暂时无法打开";
      $("reader-content").innerHTML = notice(error.message);
    }
  }
  function scrollToHash() {
    if (!location.hash) return;
    try {
      document
        .getElementById(decodeURIComponent(location.hash.slice(1)))
        ?.scrollIntoView();
    } catch {
      /* malformed hashes are ignored */
    }
  }
  function updateProgress() {
    $("to-top").hidden = scrollY < 500;
    if (!docPath()) return;
    const start = $("reader-content").offsetTop - 110;
    const end = start + $("reader-content").offsetHeight - innerHeight + 160;
    $("reading-progress").style.width =
      Math.max(
        0,
        Math.min(100, ((scrollY - start) / Math.max(1, end - start)) * 100),
      ) + "%";
  }
  // Preserve ordinary links, modifier-clicks, browser history, and shared ?doc URLs.
  document.addEventListener("click", (event) => {
    const a = event.target.closest("a[href]");
    if (
      !a ||
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      a.target === "_blank" ||
      a.hasAttribute("download")
    )
      return;
    const url = new URL(a.href);
    if (url.origin !== location.origin || url.pathname !== "/") return;
    event.preventDefault();
    const oldRoute = location.search;
    if (oldRoute !== url.search) Notebook.leaveArticle();
    document
      .querySelectorAll(".nav-more[open]")
      .forEach((el) => (el.open = false));
    history.pushState({}, "", url.pathname + url.search + url.hash);
    if ($("search-dialog").open) $("search-dialog").close();
    if (oldRoute !== location.search) {
      window.scrollTo(0, 0);
      renderRoute().then(() => {
        if (docPath()) $("reader-title").focus({ preventScroll: true });
        else if (!$("collection-view").hidden)
          $("collection-title").focus({ preventScroll: true });
        else scrollToHash();
      });
    } else {
      if (!docPath())
        document
          .querySelectorAll(".main-nav a")
          .forEach((link) =>
            link.classList.toggle(
              "active",
              link.hash === location.hash ||
                (!location.hash && link.dataset.nav === "notes"),
            ),
          );
      if (url.hash) scrollToHash();
      else window.scrollTo(0, 0);
    }
  });
  window.addEventListener("popstate", () => {
    renderRoute().then(scrollToHash);
  });
  window.addEventListener("scroll", updateProgress, { passive: true });
  $("theme-toggle").addEventListener("click", () => {
    const next =
      document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("blog-theme", next);
    } catch {}
    updateTheme();
  });
  $("search-open").addEventListener("click", openSearch);
  $("search-close").addEventListener("click", () => $("search-dialog").close());
  $("search-dialog").addEventListener("click", (event) => {
    const box = $("search-dialog").getBoundingClientRect();
    if (
      event.target === $("search-dialog") &&
      (event.clientX < box.left ||
        event.clientX > box.right ||
        event.clientY < box.top ||
        event.clientY > box.bottom)
    )
      $("search-dialog").close();
  });
  $("search-input").addEventListener("input", renderSearch);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && $("search-dialog").open) {
      event.preventDefault();
      $("search-dialog").close();
      return;
    }
    if (
      event.key === "/" &&
      !event.ctrlKey &&
      !event.metaKey &&
      !event.altKey &&
      !event.target.closest('input,textarea,[contenteditable="true"]') &&
      !document.querySelector("dialog[open]")
    ) {
      event.preventDefault();
      openSearch();
    }
  });
  $("sort-order").addEventListener("change", renderPosts);
  $("copy-link").addEventListener("click", () =>
    copyText($("copy-link"), location.href, "复制链接 ↗"),
  );
  $("to-top").addEventListener("click", () =>
    window.scrollTo({ top: 0, behavior: "smooth" }),
  );
  $("copyright-year").textContent = new Date().getFullYear();
  updateTheme();
  async function init() {
    try {
      const response = await fetch("/docs/index.json", { cache: "no-store" });
      if (!response.ok) throw new Error("文档目录加载失败");
      collect(await response.json());
      Notebook.init({ articles, href, categoryLabel, articleCard });
      renderHome();
      await renderRoute();
      if (!docPath()) requestAnimationFrame(scrollToHash);
    } catch {
      $("post-list").innerHTML = notice("文档目录暂时无法加载");
      $("spotlight").innerHTML = notice("精选文章暂时无法加载");
      if (docPath()) {
        $("home-view").hidden = true;
        $("reader-view").hidden = false;
        $("reader-title").textContent = "文章目录暂时无法加载";
        $("reader-content").innerHTML = notice("请刷新页面重试");
      }
    }
  }
  init();
})();
