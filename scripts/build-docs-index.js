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
  console.log(`已生成 ${outputFile}`);
}

main();
