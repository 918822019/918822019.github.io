#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const docsRoot = path.resolve(__dirname, 'docs');
const outputFile = path.join(docsRoot, 'index.json');

function buildTree(currentPath) {
    const stat = fs.statSync(currentPath);
    const name = path.basename(currentPath);

    if (stat.isDirectory()) {
        // Skip nbconvert asset directories like MyNotebook_files/
        if (name.endsWith('_files')) {
            return null;
        }

        const children = fs.readdirSync(currentPath)
            .filter(item => item !== 'index.json' && item !== '.DS_Store')
            .map(item => buildTree(path.join(currentPath, item)))
            .filter(Boolean)
            .sort((a, b) => {
                if (a.type !== b.type) return a.type === 'folder' ? -1 : 1;
                return a.name.localeCompare(b.name, 'zh-CN');
            });

        return {
            name,
            type: 'folder',
            children,
        };
    }

    if (stat.isFile()) {
        // Raw notebooks are converted to .md before publishing.
        if (name.endsWith('.ipynb')) {
            return null;
        }

        return {
            name,
            type: 'file',
            path: path.relative(path.resolve(__dirname), currentPath).replaceAll('\\', '/'),
        };
    }

    return null;
}

function main() {
    if (!fs.existsSync(docsRoot)) {
        console.error('错误：docs 目录不存在。');
        process.exit(1);
    }

    const tree = buildTree(docsRoot);
    fs.writeFileSync(outputFile, JSON.stringify(tree, null, 2), 'utf-8');
    console.log(`已生成 ${outputFile}`);
}

main();