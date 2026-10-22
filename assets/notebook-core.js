(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.NotebookCore = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
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
  const defaults = {
    fontSize: 16,
    lineHeight: 2,
    width: "comfortable",
    focus: false,
  };
  const clamp = (value, min, max, fallback) =>
    typeof value === "number" && Number.isFinite(value)
      ? Math.max(min, Math.min(max, value))
      : fallback;
  function preferences(value = {}) {
    return {
      fontSize: Math.round(clamp(value.fontSize, 14, 22, 16)),
      lineHeight: Math.round(clamp(value.lineHeight, 1.6, 2.4, 2) * 10) / 10,
      width: ["narrow", "comfortable", "wide"].includes(value.width)
        ? value.width
        : "comfortable",
      focus: value.focus === true,
    };
  }
  function normalize(data, paths) {
    if (
      !data ||
      data.version !== 1 ||
      !data.entries ||
      typeof data.entries !== "object" ||
      Array.isArray(data.entries)
    )
      throw new Error("不是有效的书架备份（需要 version: 1）。");
    const entries = {};
    for (const path of paths) {
      if (!Object.hasOwn(data.entries, path)) continue;
      const r = data.entries[path];
      if (!r || typeof r !== "object" || Array.isArray(r)) continue;
      entries[path] = {
        favorite: r.favorite === true,
        later: r.later === true,
        visitedAt: clamp(r.visitedAt, 0, Date.now() + 86400000, 0),
        progress: clamp(r.progress, 0, 1, 0),
        anchor: typeof r.anchor === "string" ? r.anchor.slice(0, 200) : "",
        anchorOffset: clamp(r.anchorOffset, 0, 100000, 0),
      };
    }
    return { version: 1, entries };
  }
  function merge(local, incoming) {
    const result = { version: 1, entries: { ...local.entries } };
    for (const [path, r] of Object.entries(incoming.entries)) {
      const old = result.entries[path];
      result.entries[path] = !old
        ? r
        : {
            ...(r.visitedAt > old.visitedAt ? r : old),
            favorite: old.favorite || r.favorite,
            later: old.later || r.later,
          };
    }
    return result;
  }
  function terms(query) {
    return [
      ...new Set(
        String(query).trim().toLocaleLowerCase().split(/\s+/).filter(Boolean),
      ),
    ]
      .slice(0, 8)
      .map((t) => t.slice(0, 80));
  }
  function highlight(text, query) {
    const words = terms(query).sort((a, b) => b.length - a.length);
    if (!words.length) return escape(text);
    const regex = new RegExp(
      "(" +
        words.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") +
        ")",
      "giu",
    );
    return String(text)
      .split(regex)
      .map((part, i) =>
        i % 2 ? "<mark>" + escape(part) + "</mark>" : escape(part),
      )
      .join("");
  }
  function search(index, articles, query) {
    const words = terms(query),
      byPath = new Map(index.map((item) => [item.path, item.text]));
    return articles
      .map((article) => {
        const text = byPath.get(article.path) || article.excerpt || "";
        const title = article.title.toLocaleLowerCase();
        const haystack =
          `${article.title} ${article.path} ${article.category} ${text}`.toLocaleLowerCase();
        if (!words.every((w) => haystack.includes(w))) return null;
        const score = words.reduce(
          (sum, word) =>
            sum +
            (title.includes(word) ? 20 : 0) +
            ((article.excerpt || "").toLocaleLowerCase().includes(word)
              ? 5
              : 0) +
            1,
          0,
        );
        const positions = words
          .map((w) => text.toLocaleLowerCase().indexOf(w))
          .filter((i) => i >= 0);
        const start = Math.max(
          0,
          (positions.length ? Math.min(...positions) : 0) - 45,
        );
        return {
          article,
          score,
          snippet:
            (start ? "…" : "") +
            text.slice(start, start + 180) +
            (text.length > start + 180 ? "…" : ""),
        };
      })
      .filter(Boolean)
      .sort(
        (a, b) =>
          b.score - a.score ||
          (b.article.updated || "").localeCompare(a.article.updated || "") ||
          a.article.title.localeCompare(b.article.title, "zh-CN"),
      );
  }
  return {
    escape,
    defaults,
    preferences,
    normalize,
    merge,
    terms,
    highlight,
    search,
  };
});
