import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'llama.cpp Internals',
  tagline: 'Chuỗi bài giảng phân tích chi tiết kiến trúc & hiện thực thư viện suy luận llama.cpp',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://tuandung222.github.io',
  baseUrl: '/llama-cpp-architecture-lectures/',

  organizationName: 'tuandung222',
  projectName: 'llama-cpp-architecture-lectures',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'vi',
    locales: ['vi'],
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl:
            'https://github.com/tuandung222/llama-cpp-architecture-lectures/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    metadata: [
      {name: 'robots', content: 'noindex, nofollow, noarchive, nosnippet'},
    ],
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'llama.cpp Internals',
      logo: {
        alt: 'llama.cpp Internals Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Bài học (Lectures)',
        },
        {
          href: 'https://github.com/tuandung222/llama-cpp-architecture-lectures',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Bài học',
          items: [
            {
              label: 'Bài 0: llama.cpp & Cuộc cách mạng Inference',
              to: '/docs/lesson_0_cpp_inference_revolution',
            },
            {
              label: 'Bài 2: Định dạng GGUF',
              to: '/docs/lesson_2_gguf_binary_format',
            },
            {
              label: 'Bài 9: Thực hành GGUF Reader',
              to: '/docs/lesson_9_hands_on_gguf',
            },
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/tuandung222/llama-cpp-architecture-lectures',
            },
            {
              label: 'llama.cpp GitHub',
              href: 'https://github.com/ggml-org/llama.cpp',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} llama.cpp Internals. Biên soạn bởi tuandung222. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'markdown', 'c', 'cpp'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
