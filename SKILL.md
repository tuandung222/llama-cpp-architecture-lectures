---
name: docusaurus-curriculum-builder
description: End-to-end workflow for building Vietnamese-language Docusaurus educational curricula that analyze open-source AI/ML libraries at source-code depth, with GitHub Pages deployment and privacy controls.
version: 1.0.0
author: AI Curriculum Architect
tools_required:
  - node >= 20
  - npm >= 10
  - git >= 2.30
  - gh (GitHub CLI) >= 2.40
  - cmake >= 3.20 (only if inspecting C/C++ source libraries)
---

# Docusaurus Curriculum Builder: Source-Grounded AI Library Analysis

This skill encodes the complete methodology for producing academic-grade, Vietnamese-language Docusaurus curricula that dissect open-source AI/ML libraries (such as `llama.cpp`, `vLLM`, `verl`, `ONNX Runtime`) at source-code level. It covers persona definition, privacy engineering, CI/CD automation, MDX/LaTeX pitfall avoidance, and deployment verification.

---

## 1. ĐỊNH HƯỚNG TƯ DUY & PHONG CÁCH VIẾT (Persona & Writing Persona)

### 1.1. Core Identity

Write as an **AI Expert, Deep Learning & Systems Engineering Specialist**, and a dedicated professor. The curriculum must read like an original academic lecture series in Vietnamese, not a translated blog post or marketing document.

### 1.2. Language Rules

- **Primary language**: Vietnamese.
- **English technical terms**: preserved verbatim when they are industry standard (e.g., *tensor, quantization, inference, attention, KV Cache, SIMD, AVX2, speculative decoding, perplexity, sampling*).
- **Explain before using**: every English term must be glossed in Vietnamese on first appearance.
- **Forbidden characters**: never use em dash (`—`). Use commas, colons, semicolons, or parentheses instead.
- **Section naming**: use `Phần` for course sections, `Bài` for lessons.

### 1.3. Pedagogical Flow (Mandatory Order)

Every concept must follow this sequence:

1. **Concrete system tension**: start with a real hardware or engineering problem (OOM, memory bandwidth bottleneck, GPU underutilization, latency spike).
2. **Mathematical formulation**: derive the equations using LaTeX, explain every variable.
3. **Pseudocode or algorithm**: show clean, readable pseudocode.
4. **Source-code grounding**: reference the actual file, function, and struct in the target library. **Always inspect source code before writing claims about implementation details.**
5. **Practical checklist**: actionable tuning parameters, CLI flags, configuration knobs.

### 1.4. Source Code Inspection Protocol

Before writing any claim about a library's internals:

```bash
# Clone the target library
git clone https://github.com/{LIBRARY_ORG}/{LIBRARY_REPO}.git
cd {LIBRARY_REPO}

# Read the actual headers and source files
# Verify: struct names, function signatures, enum values, constants, line numbers
# Never claim "file X implements Y" without reading file X
```

**Rule**: if you cannot verify a claim by reading source code, mark it explicitly as "based on documentation/community reports" rather than presenting it as fact.

### 1.5. Content Depth Standard

- Theory files must include derivations, not just formulas.
- Case studies must include step-by-step commands with expected output.
- Experiments must be reproducible labs with clear success criteria.
- Comparison articles must cite specific papers (authors, venue, year) and include quantitative benchmarks.

---

## 2. CÁC RÀNG BUỘC BẢO MẬT & QUYỀN RIÊNG TƯ (Security & Privacy)

### 2.1. README.md Policy

`README.md` must remain **empty (0 bytes)**. This is a deliberate privacy measure. Do not add any characters, placeholders, or whitespace.

```bash
# Verify
wc -c README.md   # Must output: 0 README.md
```

### 2.2. Search Engine Exclusion

**robots.txt** (in `static/`):

```
User-agent: *
Disallow: /
```

**Docusaurus meta tags** (in `docusaurus.config.ts`):

```typescript
scripts: [],
headTags: [
  {
    tagName: 'meta',
    attributes: {
      name: 'robots',
      content: 'noindex, nofollow, noarchive, nosnippet',
    },
  },
],
```

**Sitemap**: must remain **disabled**. Do not set `sitemap` configuration in `docusaurus.config.ts`.

**`.nojekyll`**: create an empty `static/.nojekyll` file to prevent GitHub Pages from running Jekyll.

### 2.3. Information Hygiene

Public docs must **never** contain:

- Private user instructions or hidden agent constraints.
- The fact that `README.md` is intentionally empty.
- Local absolute file paths (e.g., `/Users/{USERNAME}/repos/...`).
- Credentials, tokens, secrets, API keys, or private URLs.
- Agent coordination details or internal task IDs.

### 2.4. Git Identity

Configure git locally before any commits:

```bash
git config user.name "{GIT_USERNAME}"
git config user.email "{GIT_EMAIL}"
```

Verify the AGENT.md `Completion checklist` references the correct identity.

---

## 3. QUY TRÌNH THỰC THI & TỰ ĐỘNG HÓA (Execution Workflow)

### 3.1. Repository Initialization

```bash
# Create repo via GitHub CLI
gh repo create {GITHUB_OWNER}/{REPOSITORY_NAME} --public

# Clone locally
git clone https://github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}.git
cd {REPOSITORY_NAME}
```

### 3.2. Docusaurus Scaffolding

Do **not** use `npx create-docusaurus` inside the cloned repo (it creates a subdirectory). Instead, write files directly:

```bash
# package.json with pinned dependencies
# Dependencies: @docusaurus/core@3.10.1, react@19, react-dom@19
# Plugins: remark-math, rehype-katex, @docusaurus/theme-mermaid
# Optional: @docusaurus/faster for build speed

npm install
```

### 3.3. Content Architecture

```
docs/
├── roadmap.md                          # Main roadmap with Mermaid dependency graph
├── lesson_0_{topic}.md                 # Motivation & background
├── lesson_1_{topic}.md                 # Core theory 1
├── ...
├── lesson_N_{topic}.md                 # Hands-on practice
├── theory_deep_dive/
│   ├── roadmap_theory.md              # Theory roadmap
│   └── theory_{N}_{topic}.md          # Mathematical deep dives
├── case_studies/
│   ├── roadmap_case_studies.md        # Case study roadmap
│   └── case_{N}_{topic}.md           # Practical case studies
└── experiments_deep_dive/
    ├── roadmap_experiments.md          # Experiment roadmap (NO frontmatter)
    └── exp_{N}_{topic}.md             # Hands-on labs (NO frontmatter)
```

**Frontmatter conventions**:

| Content type | Frontmatter? | Fields |
|:---|:---|:---|
| Core lessons | Yes | `sidebar_position`, `sidebar_label` |
| Theory deep dives | Yes | `sidebar_position`, `sidebar_label` |
| Case studies | Yes | `sidebar_position`, `sidebar_label` |
| Experiment files | **No** | Bare headings only |
| Roadmap files | Yes (except experiments) | `sidebar_position`, `sidebar_label` |

**Sidebars**: use autogenerated config:

```typescript
const sidebars: SidebarsConfig = {
  tutorialSidebar: [{type: 'autogenerated', dirName: '.'}],
};
```

### 3.4. LaTeX & Mermaid in MDX

**Critical**: Docusaurus MDX parser treats `{`, `}`, and `<` as JSX syntax. This breaks LaTeX.

| Pattern | Broken | Fixed |
|:---|:---|:---|
| Block math | `\[ \text{foo} \]` | `$$\text{foo}$$` |
| Less-than in subscript | `x_{<i}` | `x_{\lt i}` |
| Angle brackets in text | `<0.1` | `dưới 0.1` or `{`<`}{0.1}` |

**Always use `$$...$$` for display math, `$...$` for inline math. Never use `\[...\]`.**

### 3.5. CI/CD: GitHub Pages Deployment

**`.github/workflows/deploy.yml`**:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{"{{"}} secrets.GITHUB_TOKEN {{"}}"}}
          publish_dir: ./build
          user_name: {GIT_USERNAME}
          user_email: {GIT_EMAIL}
```

### 3.6. Enable GitHub Pages (One-Time)

After the first successful workflow run creates the `gh-pages` branch:

```bash
echo '{"source": {"branch": "gh-pages", "path": "/"}}' | \
  gh api --method POST /repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/pages --input -
```

**Important**: GitHub Pages must be explicitly enabled via API. Even if the `gh-pages` branch exists and the workflow succeeds, the site returns HTTP 404 until Pages is enabled.

### 3.7. Build & Push Cycle

```bash
# Verify locally
npm run typecheck     # TypeScript check (must pass)
npm run build         # Static build (must succeed)

# Commit and push
git add -A
git commit -m "{COMMIT_MESSAGE}"
git push
```

The push triggers the GitHub Actions workflow which builds and deploys automatically.

---

## 4. XỬ LÝ LỖI THƯỜNG GẶP (Troubleshooting Guide)

### 4.1. MDX Compilation Errors

| Symptom | Cause | Fix |
|:---|:---|:---|
| `Unexpected character '0' before name` | `<0.1` parsed as JSX tag | Replace with Vietnamese text or MDX expression |
| `Could not parse expression with acorn` | `\text{...}` in `\[...\]` block | Use `$$...$$` instead of `\[...\]` |
| `Unexpected character '<' in expression` | `x_{<i}` in LaTeX | Use `x_{\lt i}` |
| `Expected corresponding JSX closing tag` | Unescaped `<` in table cells | Replace `<` with `&lt;` or rephrase |

### 4.2. Build Errors

| Symptom | Cause | Fix |
|:---|:---|:---|
| `npm run build` fails with KaTeX warnings | Vietnamese diacritics in math mode | Harmless warnings, build still succeeds |
| `Module not found: @docusaurus/faster` | Optional dependency missing | Remove from `package.json` or install it |
| `ENOMEM` during build | Insufficient RAM | Set `NODE_OPTIONS=--max-old-space-size=4096` |

### 4.3. GitHub Pages Errors

| Symptom | Cause | Fix |
|:---|:---|:---|
| HTTP 404 on site URL | Pages not enabled | Run the `gh api POST` command (Section 3.6) |
| HTTP 404 persists after enablement | Deployment not yet propagated | Wait 30-60 seconds after enabling |
| Old content served after push | Browser cache or CDN cache | Hard refresh (Ctrl+Shift+R) or wait for cache invalidation |
| Workflow fails on `npm ci` | `package-lock.json` out of sync | Run `npm install` locally, commit updated lockfile |

### 4.4. Git Errors

| Symptom | Cause | Fix |
|:---|:---|:---|
| `remote: Permission denied` | Token expired or insufficient scope | Regenerate PAT with `repo` scope |
| `refusing to allow integration` | Protected branch rules | Check branch protection settings |
| Wrong author in commits | Git config not set | Run `git config user.name/email` before committing |

---

## 5. TIÊU CHUẨN XÁC MINH HOÀN THÀNH (Verification Checklist)

Run all checks **before** reporting completion to the user.

### 5.1. Local Build Verification

```bash
# TypeScript type check
npm run typecheck
# Expected: exit code 0, no errors

# Static site build
npm run build
# Expected: "[SUCCESS] Generated static files in build"

# README.md size
wc -c README.md
# Expected: 0

# No em dashes in content
grep -r '—' docs/ src/
# Expected: no matches

# No absolute paths in content
grep -rE '/Users/|/home/|C:\\' docs/ src/
# Expected: no matches

# No credentials or tokens
grep -riE 'ghp_|sk-|password|secret|token' docs/ src/
# Expected: no matches (except example placeholders)
```

### 5.2. Post-Deployment Verification

```bash
# Site accessibility
curl -s -o /dev/null -w "%{http_code}" https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/
# Expected: 200

# robots.txt enforcement
curl -s https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/robots.txt
# Expected: "User-agent: *\nDisallow: /"

# noindex meta tag
curl -s https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/ | grep -o 'name="robots" content="[^"]*"'
# Expected: name="robots" content="noindex, nofollow, noarchive, nosnippet"

# README.md still empty on remote
gh api repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/contents/README.md --jq '.size'
# Expected: 0
```

### 5.3. Content Quality Verification

| Check | Method |
|:---|:---|
| All Mermaid diagrams render | Visual inspection of built pages |
| All LaTeX equations render | Visual inspection, no raw `$...$` visible |
| Frontmatter consistency | Sidebar labels match file headings |
| Source-code claims verified | Cross-reference with actual library source |
| Vietnamese reads naturally | No machine-translation artifacts |
| No broken internal links | All `[text](path.md)` links resolve |

### 5.4. Completion Summary Template

Before reporting to the user, fill in this summary:

```
Repository: https://github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}
Site URL:   https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/
Files:      {N} content files, {M} total lines
Build:      typecheck ✓ | build ✓
Privacy:    robots.txt ✓ | noindex ✓ | README 0 bytes ✓
Deploy:     HTTP 200 ✓ | gh-pages branch ✓
```

---

## Appendix A: Environment Variables Reference

| Variable | Description | Example |
|:---|:---|:---|
| `{GITHUB_OWNER}` | GitHub username or org | `tuandung222` |
| `{REPOSITORY_NAME}` | Repository name | `llama-cpp-architecture-lectures` |
| `{GIT_USERNAME}` | Git commit author name | `Tuan Dung` |
| `{GIT_EMAIL}` | Git commit author email | `user@example.com` |
| `{LIBRARY_ORG}` | Target library GitHub org | `ggerganov` |
| `{LIBRARY_REPO}` | Target library repo name | `llama.cpp` |
| `{COMMIT_MESSAGE}` | Descriptive commit message | `Initial commit: curriculum` |

## Appendix B: Docusaurus Configuration Template

Key configuration items to adapt per curriculum:

```typescript
// docusaurus.config.ts
const config = {
  title: '{LIBRARY_NAME} Internals',
  url: 'https://{GITHUB_OWNER}.github.io',
  baseUrl: '/{REPOSITORY_NAME}/',
  // KaTeX for LaTeX math
  // Mermaid for diagrams
  // Prism for code highlighting (add 'c', 'cpp', 'python', 'bash')
  // noindex meta tags in headTags
  // Vietnamese locale: 'vi'
};
```

## Appendix C: Content Type Quick Reference

| File type | Frontmatter | Math | Mermaid | Code blocks |
|:---|:---|:---|:---|:---|
| Roadmap | Yes | Optional | Yes (dependency graph) | No |
| Core lesson | Yes | Yes (LaTeX) | Yes (architecture) | Yes (C/C++/Python) |
| Theory deep dive | Yes | Heavy (derivations) | Yes (data flow) | Yes (pseudocode) |
| Case study | Yes | Moderate | Optional | Yes (bash/python) |
| Experiment | **No** | Optional | Optional | Yes (bash, step-by-step) |
| AGENT.md | **No** | No | No | Yes (shell commands) |
