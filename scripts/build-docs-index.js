#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const repoRoot = path.resolve(__dirname, "..");
const docsRoot = path.join(repoRoot, "docs");
const outputFile = path.join(docsRoot, "index.json");

function articleMetadata(filePath) {
  const markdown = fs.readFileSync(filePath, "utf8");
  const prose = markdown
    .replace(/```[\s\S]*?```/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "");
  const titleMatch = prose.match(/^#\s+(.+)$/m);
  const title = (titleMatch ? titleMatch[1] : path.basename(filePath, ".md"))
    .replace(/[*`]/g, "")
    .replace(/^\d+[.、]\s*/, "")
    .trim();
  const paragraphs = prose
    .split(/\r?\n/)
    .filter((line) => !/^ {4}|^\t/.test(line))
    .map((line) => line.trim().replace(/^[-*+]\s+/, ""))
    .filter((line) => line.length > 25 && !/^[#>|$\[]|^\d+[.)]/.test(line));
  const excerpt = (paragraphs[0] || `阅读「${title}」的完整学习与实践记录。`)
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*`]/g, "")
    .slice(0, 120);
  let updated = null;
  try {
    updated =
      execFileSync(
        "git",
        ["log", "-1", "--format=%cs", "--", path.relative(repoRoot, filePath)],
        {
          cwd: repoRoot,
          encoding: "utf8",
          stdio: ["ignore", "pipe", "ignore"],
        },
      ).trim() || null;
  } catch {
    /* Dates are optional when building without Git history. */
  }
  return {
    title,
    excerpt,
    readingMinutes: Math.max(
      1,
      Math.ceil(prose.replace(/\s/g, "").length / 450),
    ),
    updated,
  };
}

function buildTree(currentPath) {
  const stat = fs.statSync(currentPath);
  const name = path.basename(currentPath);

  if (stat.isDirectory()) {
    // Skip nbconvert asset directories like MyNotebook_files/
    if (name.endsWith("_files")) {
      return null;
    }

    const children = fs
      .readdirSync(currentPath)
      .filter((item) => item !== "index.json" && item !== ".DS_Store")
      .map((item) => buildTree(path.join(currentPath, item)))
      .filter(Boolean)
      .sort((a, b) => {
        if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
        return a.name.localeCompare(b.name, "zh-CN");
      });

    return {
      name,
      type: "folder",
      children,
    };
  }

  if (stat.isFile()) {
    // Raw notebooks are converted to .md before publishing.
    if (name.endsWith(".ipynb")) {
      return null;
    }

    return {
      name,
      type: "file",
      path: path.relative(repoRoot, currentPath).replaceAll("\\", "/"),
      ...(name.endsWith(".md") ? articleMetadata(currentPath) : {}),
    };
  }

  return null;
}

function main() {
  if (!fs.existsSync(docsRoot)) {
    console.error("错误：docs 目录不存在。");
    process.exit(1);
  }

  const tree = buildTree(docsRoot);
  fs.writeFileSync(outputFile, JSON.stringify(tree, null, 2), "utf-8");
  buildPublicIndexes(tree);
  console.log(`已生成 ${outputFile}`);
}

function buildPublicIndexes(tree) {
  const files = [];
  function visit(node) {
    if (
      node.type === "file" &&
      node.path.endsWith(".md") &&
      !/\/README\.md$/i.test(node.path)
    )
      files.push(node);
    (node.children || []).forEach(visit);
  }
  visit(tree);
  const output = path.join(repoRoot, "assets", "content");
  fs.mkdirSync(output, { recursive: true });
  const search = files.map((file) => ({
    path: file.path,
    title: file.title,
    text: fs
      .readFileSync(path.join(repoRoot, file.path), "utf8")
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/<[^>]*>/g, " ")
      .replace(/```[^\n]*\n/g, "\n")
      .replace(/[#*`>|]/g, " ")
      .replace(/\s+/g, " ")
      .trim(),
  }));
  fs.writeFileSync(
    path.join(output, "search.json"),
    JSON.stringify({ version: 1, articles: search }, null, 2),
  );
  const series = JSON.parse(
    fs.readFileSync(path.join(__dirname, "series.json"), "utf8"),
  );
  const validPaths = new Set(files.map((f) => f.path));
  const seriesIds = new Set();
  for (const item of series) {
    if (seriesIds.has(item.id) || !/^[a-z0-9-]+$/.test(item.id))
      throw new Error(`Invalid series ID: ${item.id}`);
    seriesIds.add(item.id);
    for (const file of item.paths)
      if (!validPaths.has(file))
        throw new Error(`Missing series article: ${file}`);
    if (new Set(item.paths).size !== item.paths.length)
      throw new Error(`Duplicate article in series: ${item.id}`);
  }
  fs.writeFileSync(
    path.join(output, "series.json"),
    JSON.stringify(series, null, 2),
  );
  const xml = (value) =>
    String(value).replace(
      /[<>&"']/g,
      (c) =>
        ({
          "<": "&lt;",
          ">": "&gt;",
          "&": "&amp;",
          '"': "&quot;",
          "'": "&apos;",
        })[c],
    );
  const site = "https://918822019.github.io";
  const sorted = [...files].sort(
    (a, b) =>
      (b.updated || "").localeCompare(a.updated || "") ||
      a.path.localeCompare(b.path),
  );
  const items = sorted.map((file) => {
    const link = site + "/?doc=" + encodeURIComponent(file.path);
    return `    <item>\n      <title>${xml(file.title)}</title>\n      <link>${xml(link)}</link>\n      <guid isPermaLink="true">${xml(link)}</guid>\n      <description>${xml(file.excerpt)}</description>\n${file.updated ? "      <pubDate>" + new Date(file.updated + "T00:00:00+08:00").toUTCString() + "</pubDate>\n" : ""}    </item>`;
  });
  fs.writeFileSync(
    path.join(repoRoot, "feed.xml"),
    `<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n  <channel>\n    <title>918822019 · 技术手记</title>\n    <link>${site}/</link>\n    <description>AI、数学与工程实践的学习记录，按文章更新日期订阅。</description>\n    <language>zh-cn</language>\n    <atom:link href="${site}/feed.xml" rel="self" type="application/rss+xml"/>\n${items.join("\n")}\n  </channel>\n</rss>\n`,
  );
  console.log(
    `已生成全文索引、${series.length} 个系列和 ${files.length} 篇 RSS 条目`,
  );
}

main();
