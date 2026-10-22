# Local browser dependencies

These browser distributions are committed so article reading does not depend on a third-party CDN.

| Package | Version | Source |
| --- | --- | --- |
| marked | 18.1.1 | https://www.npmjs.com/package/marked |
| DOMPurify | 3.4.17 | https://www.npmjs.com/package/dompurify |
| KaTeX | 0.19.0 | https://www.npmjs.com/package/katex |
| Prism | 1.30.0 | https://www.npmjs.com/package/prismjs |

Licenses are included in each package directory. Markdown is parsed with marked, sanitized with DOMPurify, and then enhanced with KaTeX. To update, copy the corresponding browser builds and licenses from the npm package; KaTeX fonts must stay alongside its CSS under `katex/fonts/`.

`prism/prism.min.js` combines the core and markup, CSS, clike, JavaScript, TypeScript, Python, Bash, JSON, C, C++, SQL, PowerShell, YAML and INI language components. Automatic highlighting is disabled; the article reader highlights its rendered code blocks explicitly.
