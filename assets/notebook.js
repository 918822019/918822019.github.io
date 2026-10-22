(() => {
  "use strict";
  const $ = (id) => document.getElementById(id),
    core = window.NotebookCore,
    esc = core.escape;
  const stateKey = "field-notes-library-v1",
    prefsKey = "field-notes-reading-v1";
  let articles = [],
    paths = [],
    series = [],
    seriesError = false,
    state = { version: 1, entries: {} },
    prefs = { ...core.defaults };
  let api,
    current = null,
    resume = null,
    saveTimer,
    statusTimer,
    imageList = [],
    imageIndex = 0,
    zoom = 1,
    searchPromise;
  let storageWarned = false,
    collectionRequest = 0;
  let seriesPromise = Promise.resolve();
  const record = (path) =>
    state.entries[path] || {
      favorite: false,
      later: false,
      visitedAt: 0,
      progress: 0,
      anchor: "",
      anchorOffset: 0,
    };
  function toast(message) {
    $("feature-status").textContent = message;
    $("feature-status").hidden = false;
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => {
      $("feature-status").hidden = true;
    }, 4200);
  }
  function persist() {
    try {
      localStorage.setItem(stateKey, JSON.stringify(state));
    } catch {
      if (!storageWarned) {
        toast("浏览器未能保存书架。当前操作仍可使用，请导出备份。");
        storageWarned = true;
      }
    }
  }
  function applyPreferences() {
    const root = document.documentElement;
    root.style.setProperty("--reader-font-size", prefs.fontSize + "px");
    root.style.setProperty("--reader-line-height", String(prefs.lineHeight));
    root.style.setProperty(
      "--reading-width",
      { narrow: "640px", comfortable: "800px", wide: "900px" }[prefs.width],
    );
    document.body.classList.toggle("focus-reading", !!current && prefs.focus);
    $("focus-toggle").textContent = prefs.focus ? "退出专注" : "专注阅读";
    $("focus-toggle").setAttribute("aria-pressed", String(prefs.focus));
    $("reading-size").value = prefs.fontSize;
    $("size-value").textContent = prefs.fontSize + "px";
    $("reading-spacing").value = prefs.lineHeight;
    $("spacing-value").textContent = prefs.lineHeight.toFixed(1);
    $("reading-width").value = prefs.width;
  }
  function savePreferences() {
    applyPreferences();
    try {
      localStorage.setItem(prefsKey, JSON.stringify(prefs));
    } catch {
      toast("阅读设置已应用，本次无法持久保存。");
    }
  }
  function updateButtons() {
    if (!current) return;
    const r = record(current.path);
    $("favorite-toggle").textContent = r.favorite ? "★ 已收藏" : "☆ 收藏";
    $("favorite-toggle").setAttribute("aria-pressed", String(r.favorite));
    $("later-toggle").textContent = r.later ? "✓ 已加入稍后阅读" : "◷ 稍后阅读";
    $("later-toggle").setAttribute("aria-pressed", String(r.later));
  }
  function toggle(path, property) {
    if (!paths.includes(path)) return;
    const r = { ...record(path) };
    r[property] = !r[property];
    state.entries[path] = r;
    persist();
    updateButtons();
    if (!$("collection-view").hidden)
      renderCollection("library", new URLSearchParams(location.search));
    toast(
      property === "favorite"
        ? r.favorite
          ? "已加入收藏"
          : "已取消收藏"
        : r.later
          ? "已加入稍后阅读"
          : "已移出稍后阅读",
    );
  }
  function progress() {
    const content = $("reader-content"),
      top = content.getBoundingClientRect().top + scrollY;
    return Math.max(
      0,
      Math.min(
        1,
        (scrollY - top + 110) /
          Math.max(1, content.offsetHeight - innerHeight + 160),
      ),
    );
  }
  function saveReading() {
    clearTimeout(saveTimer);
    if (!current || resume || $("reader-view").hidden) return;
    const headings = [...$("reader-content").querySelectorAll("h2,h3")];
    const nearest = headings
      .filter((h) => h.getBoundingClientRect().top < 130)
      .at(-1);
    state.entries[current.path] = {
      ...record(current.path),
      visitedAt: Date.now(),
      progress: progress(),
      anchor: nearest?.id || "",
      anchorOffset: nearest
        ? Math.max(0, 110 - nearest.getBoundingClientRect().top)
        : 0,
    };
    persist();
  }
  function leaveArticle() {
    saveReading();
    current = null;
    resume = null;
    clearTimeout(saveTimer);
    applyPreferences();
  }
  async function search(query) {
    if (!searchPromise)
      searchPromise = fetch("/assets/content/search.json", {
        cache: "no-store",
      })
        .then((r) => {
          if (!r.ok) throw new Error();
          return r.json();
        })
        .then((data) => {
          if (!Array.isArray(data.articles)) throw new Error();
          return data.articles;
        })
        .catch(() => {
          searchPromise = null;
          return null;
        });
    const index = await searchPromise;
    return {
      matches: core.search(
        index || [],
        articles
          .filter((a) => !/\/README\.md$/i.test(a.path))
          .map((a) => ({ ...a, category: api.categoryLabel(a.category) })),
        query,
      ),
      fullText: !!index,
    };
  }
  async function renderCollection(view, params) {
    const request = ++collectionRequest;
    $("collection-tools").innerHTML = "";
    $("collection-content").innerHTML = "";
    const title =
      view === "archive"
        ? "把积累，交给时间。"
        : view === "series"
          ? "沿着一条线索，读下去。"
          : "我的书架";
    $("collection-title").textContent = title;
    document.title = title + " · 918822019";
    $("collection-eyebrow").textContent =
      view === "archive"
        ? "THE ARCHIVE"
        : view === "series"
          ? "LEARNING PATHS"
          : "YOUR READING SPACE";
    $("collection-description").textContent =
      view === "archive"
        ? "按文章最近更新的年月归档，也可以通过 RSS 订阅更新。"
        : view === "series"
          ? "按主题组织的阅读路线，循序渐进，也可以从感兴趣的一篇开始。"
          : "收藏、稍后阅读和阅读进度保存在当前浏览器。导出备份后，可在其他设备手动导入。";
    if (view === "library") renderLibrary(params.get("tab"));
    if (view === "archive") renderArchive(params.get("year"));
    if (view === "series") {
      $("collection-content").innerHTML =
        '<p class="loading-placeholder">正在加载阅读路线…</p>';
      await seriesPromise;
      if (request !== collectionRequest) return;
      renderSeries(params.get("series"));
    }
  }
  function renderLibrary(tab) {
    tab = ["favorites", "later", "history"].includes(tab) ? tab : "favorites";
    const tabs = [
      ["favorites", "我的收藏"],
      ["later", "稍后阅读"],
      ["history", "最近阅读"],
    ];
    const matches = (path) =>
      tab === "favorites"
        ? record(path).favorite
        : tab === "later"
          ? record(path).later
          : record(path).visitedAt > 0;
    const files = articles
      .filter((a) => matches(a.path))
      .sort(
        (a, b) =>
          record(b.path).visitedAt - record(a.path).visitedAt ||
          a.title.localeCompare(b.title, "zh-CN"),
      );
    $("collection-tools").innerHTML =
      `<div class="collection-toolbar"><nav class="shelf-tabs" aria-label="书架分类">${tabs.map(([id, label]) => `<a href="/?view=library&tab=${id}" class="shelf-tab${id === tab ? " active" : ""}" ${id === tab ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav><div class="shelf-actions"><button class="tool-button" data-action="export">导出备份</button><button class="tool-button" data-action="import">导入备份</button>${tab === "history" && files.length ? '<button class="plain-button" data-action="clear-history">清空历史</button>' : ""}</div></div>`;
    $("collection-content").innerHTML = files.length
      ? `<p class="shelf-count">${files.length} 篇手记</p><div class="shelf-list">${files
          .map((a) => {
            const r = record(a.path),
              percentage = Math.round(r.progress * 100);
            return `<article class="shelf-row"><div><span class="post-category">${esc(api.categoryLabel(a.category))}</span><h2><a href="${api.href(a.path)}">${esc(a.title)}</a></h2><p>${r.visitedAt ? `上次阅读 ${new Date(r.visitedAt).toLocaleDateString("zh-CN")} · ${percentage >= 95 ? "已读至结尾" : `已读 ${percentage}%`}` : "还未开始阅读"} · 约 ${a.readingMinutes} 分钟</p><div class="shelf-progress" aria-label="阅读进度 ${percentage}%"><span style="width:${percentage}%"></span></div></div><div class="shelf-row-actions"><a class="text-link" href="${api.href(a.path)}">${r.progress > 0.03 ? "继续阅读" : "开始阅读"} ↗</a><button class="plain-button" data-action="${tab === "history" ? "remove-history" : tab === "favorites" ? "favorite" : "later"}" data-path="${esc(a.path)}">${tab === "history" ? "移除记录" : "移出列表"}</button></div></article>`;
          })
          .join("")}</div>`
      : `<div class="shelf-empty"><span aria-hidden="true">${tab === "favorites" ? "☆" : tab === "later" ? "◷" : "↺"}</span><h2>${tab === "favorites" ? "给值得再读的文章留个位置" : tab === "later" ? "把想读的文章，留给下一次" : "阅读的足迹会留在这里"}</h2><p>打开任意手记，使用文章上方的阅读工具。</p><a class="text-link" href="/#notes">去发现一篇手记 ↗</a></div>`;
  }
  function renderArchive(year) {
    const files = articles
      .filter((a) => !/\/README\.md$/i.test(a.path))
      .sort(
        (a, b) =>
          (b.updated || "").localeCompare(a.updated || "") ||
          a.title.localeCompare(b.title, "zh-CN"),
      );
    const years = [
      ...new Set(files.map((a) => a.updated?.slice(0, 4)).filter(Boolean)),
    ];
    if (!years.includes(year)) year = "all";
    $("collection-tools").innerHTML =
      `<div class="collection-toolbar"><nav class="shelf-tabs" aria-label="归档年份"><a class="shelf-tab${year === "all" ? " active" : ""}" href="/?view=archive">全部年份</a>${years.map((y) => `<a class="shelf-tab${year === y ? " active" : ""}" href="/?view=archive&year=${y}">${y}</a>`).join("")}</nav><div class="shelf-actions"><button class="tool-button" data-action="rss-copy">复制 RSS 地址</button><a class="text-link" href="/feed.xml" target="_blank">打开 RSS ↗</a></div></div>`;
    const groups = new Map();
    for (const file of files.filter(
      (a) => year === "all" || a.updated?.startsWith(year),
    )) {
      const month = file.updated?.slice(0, 7) || "日期未记录";
      if (!groups.has(month)) groups.set(month, []);
      groups.get(month).push(file);
    }
    $("collection-content").innerHTML = [...groups]
      .map(
        ([month, entries]) =>
          `<section class="archive-month"><h2>${esc(month.replace("-", " / "))}<small>${entries.length} 篇</small></h2><ol>${entries.map((a) => `<li><time datetime="${a.updated || ""}">${a.updated?.slice(8) || "—"}</time><a href="${api.href(a.path)}">${esc(a.title)}</a><span>${esc(api.categoryLabel(a.category))}</span></li>`).join("")}</ol></section>`,
      )
      .join("");
  }
  function renderSeries(selected) {
    if (seriesError) {
      $("collection-content").innerHTML =
        '<p class="notice">系列目录暂时无法加载，请刷新重试。你仍可从首页阅读全部文章。</p>';
      return;
    }
    const selectedSeries = series.find((s) => s.id === selected),
      list = selectedSeries ? [selectedSeries] : series;
    $("collection-tools").innerHTML = selectedSeries
      ? '<a class="text-link" href="/?view=series">← 返回全部系列</a>'
      : "";
    $("collection-content").innerHTML = `<div class="series-grid">${list
      .map(
        (s, i) =>
          `<section class="series-card"><p class="eyebrow">PATH ${String(i + 1).padStart(2, "0")} / ${s.paths.length} NOTES</p><h2><a href="/?view=series&series=${s.id}">${esc(s.title)}</a></h2><p>${esc(s.description)}</p><ol>${s.paths
            .map((path) => {
              const a = articles.find((a) => a.path === path);
              return a
                ? `<li><a href="${api.href(path)}">${esc(a.title)}</a><span>${record(path).progress >= 0.95 ? "已读" : `约 ${a.readingMinutes} 分钟`}</span></li>`
                : "";
            })
            .join(
              "",
            )}</ol><a class="text-link" href="${api.href(s.paths.find((p) => record(p).progress < 0.95) || s.paths[0])}">沿着这条路线阅读 ↗</a></section>`,
      )
      .join("")}</div>`;
  }
  async function showSeriesContext(article) {
    $("series-context").innerHTML = "";
    $("series-neighbors").innerHTML = "";
    await seriesPromise;
    if (current?.path !== article.path) return;
    const s = series.find((s) => s.paths.includes(article.path));
    if (!s) return;
    const i = s.paths.indexOf(article.path);
    $("series-context").innerHTML =
      `<div class="series-context"><a href="/?view=series&series=${s.id}">${esc(s.title)} ↗</a><span>第 ${i + 1} / ${s.paths.length} 篇</span></div>`;
    $("series-neighbors").innerHTML = [
      [i - 1, "上一篇"],
      [i + 1, "下一篇"],
    ]
      .map(([idx, label]) => {
        const a = articles.find((a) => a.path === s.paths[idx]);
        return a
          ? `<a href="${api.href(a.path)}"><small>${label}</small><strong>${esc(a.title)}</strong><span aria-hidden="true">${label === "上一篇" ? "←" : "→"}</span></a>`
          : "<span></span>";
      })
      .join("");
  }
  function setupArticle(article) {
    current = article;
    const previous = { ...record(article.path) };
    resume =
      previous.visitedAt && previous.progress > 0.03 && !location.hash
        ? previous
        : null;
    state.entries[article.path] = { ...previous, visitedAt: Date.now() };
    persist();
    $("resume-banner").hidden = !resume;
    if (resume)
      $("resume-message").textContent =
        `上次读到 ${Math.round(previous.progress * 100)}%，从那里继续？`;
    applyPreferences();
    updateButtons();
    showSeriesContext(article);
    imageList = [...$("reader-content").querySelectorAll("img")];
    imageList.forEach((img, index) => {
      img.tabIndex = 0;
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", "放大图片：" + (img.alt || "文章插图"));
      img.addEventListener("click", (e) => {
        e.preventDefault();
        openImage(index);
      });
      img.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openImage(index);
        }
      });
    });
  }
  function continueReading() {
    if (!resume) return;
    const saved = resume;
    resume = null;
    $("resume-banner").hidden = true;
    requestAnimationFrame(() => {
      const anchor = saved.anchor && document.getElementById(saved.anchor);
      const content = $("reader-content");
      const target = anchor
        ? anchor.getBoundingClientRect().top +
          scrollY -
          110 +
          saved.anchorOffset
        : content.getBoundingClientRect().top +
          scrollY -
          110 +
          saved.progress *
            Math.max(1, content.offsetHeight - innerHeight + 160);
      window.scrollTo({ top: target, behavior: "instant" });
      saveReading();
    });
  }
  function openImage(index) {
    imageIndex = index;
    zoom = 1;
    renderImage();
    $("image-dialog").showModal();
  }
  function renderImage() {
    const image = imageList[imageIndex];
    if (!image) return;
    $("image-full").src = image.currentSrc || image.src;
    $("image-full").alt = image.alt;
    $("image-full").style.width = zoom * 100 + "%";
    $("image-full").classList.toggle("zoomed", zoom > 1);
    $("image-counter").textContent =
      `${imageIndex + 1} / ${imageList.length} · ${Math.round(zoom * 100)}%`;
    $("image-caption").textContent = image.alt || "文章插图";
    $("image-prev").disabled = imageIndex === 0;
    $("image-next").disabled = imageIndex === imageList.length - 1;
    $("image-zoom-in").disabled = zoom >= 3;
    $("image-zoom-out").disabled = zoom <= 1;
  }
  function changeImage(direction) {
    imageIndex = Math.max(
      0,
      Math.min(imageList.length - 1, imageIndex + direction),
    );
    zoom = 1;
    renderImage();
    $("image-stage").scrollTo(0, 0);
  }
  function exportLibrary() {
    saveReading();
    const blob = new Blob(
      [
        JSON.stringify(
          { ...state, exportedAt: new Date().toISOString() },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download =
      "918822019-reading-backup-" +
      new Date().toISOString().slice(0, 10) +
      ".json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("书架备份已导出");
  }
  async function importLibrary(file) {
    if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024)
        throw new Error("文件超过 2 MB，请选择本站导出的 JSON 备份。");
      const parsed = JSON.parse(await file.text()),
        incoming = core.normalize(parsed, paths);
      state = core.merge(state, incoming);
      persist();
      renderCollection("library", new URLSearchParams(location.search));
      toast(
        `已合并 ${Object.keys(incoming.entries).length} 篇现有文章的记录；原收藏会保留。`,
      );
    } catch (error) {
      toast(
        error instanceof SyntaxError
          ? "文件不是有效的 JSON，书架未更改。"
          : error.message,
      );
    }
    $("library-import").value = "";
  }
  function clearHistory(path) {
    if (path && !paths.includes(path)) return;
    for (const p of path ? [path] : Object.keys(state.entries))
      state.entries[p] = {
        ...record(p),
        visitedAt: 0,
        progress: 0,
        anchor: "",
        anchorOffset: 0,
      };
    persist();
    renderCollection("library", new URLSearchParams(location.search));
    toast("阅读记录已移除，收藏与稍后阅读保留。");
  }
  function bind() {
    const dismissResume = () => {
      if (current && resume) {
        resume = null;
        $("resume-banner").hidden = true;
      }
    };
    window.addEventListener("wheel", dismissResume, { passive: true });
    window.addEventListener("touchmove", dismissResume, { passive: true });
    document.addEventListener("keydown", (event) => {
      if (
        !document.querySelector("dialog[open]") &&
        !event.target.closest("input,textarea,select,button") &&
        [
          "PageDown",
          "PageUp",
          "ArrowDown",
          "ArrowUp",
          "Home",
          "End",
          " ",
        ].includes(event.key)
      )
        dismissResume();
    });
    document.addEventListener("click", (event) => {
      if (event.target.closest("#article-toc a")) dismissResume();
      if (!event.target.closest(".nav-more"))
        document
          .querySelectorAll(".nav-more[open]")
          .forEach((menu) => (menu.open = false));
    });
    $("favorite-toggle").addEventListener(
      "click",
      () => current && toggle(current.path, "favorite"),
    );
    $("later-toggle").addEventListener(
      "click",
      () => current && toggle(current.path, "later"),
    );
    $("settings-open").addEventListener("click", () =>
      $("reading-settings").showModal(),
    );
    $("focus-toggle").addEventListener("click", () => {
      prefs.focus = !prefs.focus;
      savePreferences();
    });
    $("reading-size").addEventListener("input", (e) => {
      prefs.fontSize = Number(e.target.value);
      savePreferences();
    });
    $("reading-spacing").addEventListener("input", (e) => {
      prefs.lineHeight = Number(e.target.value);
      savePreferences();
    });
    $("reading-width").addEventListener("change", (e) => {
      prefs.width = e.target.value;
      savePreferences();
    });
    $("settings-reset").addEventListener("click", () => {
      prefs = { ...core.defaults };
      savePreferences();
    });
    $("resume-reading").addEventListener("click", continueReading);
    $("restart-reading").addEventListener("click", () => {
      resume = null;
      $("resume-banner").hidden = true;
      scrollTo({ top: 0, behavior: "instant" });
      saveReading();
    });
    $("library-import").addEventListener("change", (e) =>
      importLibrary(e.target.files[0]),
    );
    $("image-prev").addEventListener("click", () => changeImage(-1));
    $("image-next").addEventListener("click", () => changeImage(1));
    $("image-zoom-in").addEventListener("click", () => {
      zoom = Math.min(3, zoom + 0.5);
      renderImage();
    });
    $("image-zoom-out").addEventListener("click", () => {
      zoom = Math.max(1, zoom - 0.5);
      renderImage();
    });
    $("image-reset").addEventListener("click", () => {
      zoom = 1;
      renderImage();
      $("image-stage").scrollTo(0, 0);
    });
    document
      .querySelectorAll("[data-close-dialog]")
      .forEach((button) =>
        button.addEventListener("click", () =>
          $(button.dataset.closeDialog).close(),
        ),
      );
    for (const id of ["image-dialog", "reading-settings"])
      $(id).addEventListener("click", (e) => {
        const rect = $(id).getBoundingClientRect();
        if (
          e.target === $(id) &&
          (e.clientX < rect.left ||
            e.clientX > rect.right ||
            e.clientY < rect.top ||
            e.clientY > rect.bottom)
        )
          $(id).close();
      });
    document.addEventListener("keydown", (e) => {
      if ($("image-dialog").open) {
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          changeImage(-1);
        }
        if (e.key === "ArrowRight") {
          e.preventDefault();
          changeImage(1);
        }
      }
    });
    $("collection-tools").addEventListener("click", actions);
    $("collection-content").addEventListener("click", actions);
    window.addEventListener(
      "scroll",
      () => {
        if (current && !resume) {
          clearTimeout(saveTimer);
          saveTimer = setTimeout(saveReading, 650);
        }
      },
      { passive: true },
    );
    window.addEventListener("pagehide", saveReading);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) saveReading();
    });
    window.addEventListener("storage", (event) => {
      if (event.key === stateKey) {
        try {
          state = event.newValue
            ? core.normalize(JSON.parse(event.newValue), paths)
            : { version: 1, entries: {} };
          updateButtons();
          if (
            !$("collection-view").hidden &&
            new URLSearchParams(location.search).get("view") === "library"
          )
            renderCollection("library", new URLSearchParams(location.search));
        } catch {}
      }
    });
  }
  function actions(event) {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "export") exportLibrary();
    if (action === "import") $("library-import").click();
    if (action === "favorite" || action === "later")
      toggle(button.dataset.path, action);
    if (action === "remove-history") clearHistory(button.dataset.path);
    if (
      action === "clear-history" &&
      confirm("清空此浏览器中的阅读历史和进度？收藏与稍后阅读将保留。")
    )
      clearHistory();
    if (action === "rss-copy")
      navigator.clipboard
        .writeText("https://918822019.github.io/feed.xml")
        .then(() => toast("RSS 地址已复制"))
        .catch(() => toast("复制失败，请打开 RSS 页面并复制地址。"));
  }
  function init(options) {
    api = options;
    articles = options.articles;
    paths = articles.map((a) => a.path);
    try {
      const saved = localStorage.getItem(stateKey);
      if (saved) state = core.normalize(JSON.parse(saved), paths);
    } catch {
      toast("保存的书架记录暂时无法读取，阅读功能仍可使用。");
    }
    try {
      prefs = core.preferences(
        JSON.parse(localStorage.getItem(prefsKey) || "{}"),
      );
    } catch {}
    seriesPromise = fetch("/assets/content/series.json", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((data) => {
        series = data.filter(
          (s) =>
            Array.isArray(s.paths) && s.paths.every((p) => paths.includes(p)),
        );
      })
      .catch(() => {
        seriesError = true;
      });
    bind();
    applyPreferences();
  }
  window.Notebook = {
    init,
    search,
    renderCollection,
    setupArticle,
    leaveArticle,
    saveReading,
    toast,
  };
})();
