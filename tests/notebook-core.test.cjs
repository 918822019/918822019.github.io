const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../assets/notebook-core.js");
const path = "docs/example.md";

test("full-text search finds body-only terms, uses AND matching and ranks titles first", () => {
  const articles = [
    { path, title: "Model notes", excerpt: "overview", category: "infra" },
    { path: "docs/b.md", title: "needle", excerpt: "other", category: "infra" },
  ];
  const index = [
    { path, text: "A hidden needle and GPU computation" },
    { path: "docs/b.md", text: "needle is here" },
  ];
  assert.equal(
    core.search(index, articles, "needle")[0].article.title,
    "needle",
  );
  assert.equal(core.search(index, articles, "needle GPU").length, 1);
  assert.match(core.search(index, articles, "needle GPU")[0].snippet, /GPU/);
  assert.equal(core.search(index, articles, "missing").length, 0);
});
test("snippets highlight literal queries while escaping HTML and regex metacharacters", () => {
  assert.equal(
    core.highlight("<img onerror=x> C++ [test]", "C++"),
    "&lt;img onerror=x&gt; <mark>C++</mark> [test]",
  );
  assert.equal(core.highlight("a.b aXb", "a.b"), "<mark>a.b</mark> aXb");
  assert.equal(core.highlight("plain", ""), "plain");
});
test("import validates schema, ignores unknown paths and bounds position values", () => {
  assert.throws(() => core.normalize({ version: 2, entries: {} }, [path]));
  assert.throws(() => core.normalize({ version: 1, entries: [] }, [path]));
  const data = JSON.parse(
    '{"version":1,"entries":{"__proto__":{"polluted":true},"docs/example.md":{"favorite":"true","later":true,"progress":9,"visitedAt":"bad","anchorOffset":-50},"docs/deleted.md":{"favorite":true}}}',
  );
  const state = core.normalize(data, [path]);
  assert.deepEqual(Object.keys(state.entries), [path]);
  assert.equal(state.entries[path].favorite, false);
  assert.equal(state.entries[path].later, true);
  assert.equal(state.entries[path].progress, 1);
  assert.equal(state.entries[path].visitedAt, 0);
  assert.equal(state.entries[path].anchorOffset, 0);
  assert.equal({}.polluted, undefined);
});
test("merge preserves existing saves and keeps the newer reading progress", () => {
  const local = {
    version: 1,
    entries: {
      [path]: { favorite: true, later: false, visitedAt: 200, progress: 0.6 },
    },
  };
  const incoming = {
    version: 1,
    entries: {
      [path]: { favorite: false, later: true, visitedAt: 100, progress: 0.1 },
    },
  };
  const merged = core.merge(local, incoming);
  assert.equal(merged.entries[path].favorite, true);
  assert.equal(merged.entries[path].later, true);
  assert.equal(merged.entries[path].progress, 0.6);
  assert.equal(local.entries[path].later, false);
});
test("reading preferences reject malformed values", () => {
  assert.deepEqual(
    core.preferences({
      fontSize: 100,
      lineHeight: 0,
      width: "invalid",
      focus: "yes",
    }),
    { fontSize: 22, lineHeight: 1.6, width: "comfortable", focus: false },
  );
  assert.equal(core.preferences({ fontSize: NaN }).fontSize, 16);
});
