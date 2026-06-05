import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero custom-hero', styles.heroBanner)}>
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'inline-flex', padding: '4px 12px', background: 'rgba(139, 92, 246, 0.15)', color: '#a78bfa', borderRadius: '100px', fontSize: '0.85rem', fontWeight: 600, border: '1px solid rgba(139, 92, 246, 0.25)', marginBottom: '1.5rem' }}>
          DOCUMENTATION &amp; PRACTICAL LLM INFERENCE COURSE
        </div>
        <Heading as="h1" className="hero__title" style={{ fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(135deg, #fff 30%, #a78bfa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '1.5rem' }}>
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle" style={{ maxWidth: '750px', margin: '0 auto 2.5rem auto', fontSize: '1.2rem', lineHeight: '1.6', opacity: 0.85 }}>
          {siteConfig.tagline}
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', boxShadow: '0 4px 14px rgba(139, 92, 246, 0.4)', transition: 'all 0.3s ease' }}
            to="/docs/roadmap">
            Bắt đầu học ngay 🚀
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            style={{ padding: '0.8rem 2rem', fontSize: '1.05rem', fontWeight: 600, borderRadius: '8px', marginLeft: '1rem', border: '1px solid var(--card-border)' }}
            href="https://github.com/tuandung222/llama-cpp-architecture-lectures">
            View on GitHub 🛠️
          </Link>
        </div>
      </div>
    </header>
  );
}

interface FeatureItem {
  title: string;
  badge: string;
  description: string;
}

const CorePillars: FeatureItem[] = [
  {
    title: 'GGML Tensor Library',
    badge: 'Core Engine',
    description: 'Thư viện tensor bằng C thuần túy, không phụ thuộc, hỗ trợ SIMD trên CPU (AVX2/AVX512/NEON), CUDA, Metal và Vulkan. Arena-based memory allocation và computation graph tối ưu cho inference.',
  },
  {
    title: 'GGUF Binary Format',
    badge: 'Data Format',
    description: 'Định dạng nhị phân tối ưu cho SIMD với 32-byte alignment, hỗ trợ 30+ loại quantization (Q4_0 đến IQ4_NL), hệ thống metadata key-value linh hoạt cho việc lưu trữ và tải mô hình.',
  },
  {
    title: 'Multi-Backend Inference',
    badge: 'Hardware Abstraction',
    description: 'Kiến trúc backend trừu tượng hóa phần cứng: CPU (AVX2/AVX512/NEON), CUDA partial offload, Metal MPS cho Apple Silicon, Vulkan cross-platform. Hỗ trợ từ Raspberry Pi đến datacenter GPU.',
  },
];

interface LectureItem {
  number: string;
  title: string;
  desc: string;
  path: string;
  category: 'Background' | 'Core Theory' | 'Deep Dive Code' | 'Optimization' | 'Practice';
}

const Lectures: LectureItem[] = [
  {
    number: 'Bài 0',
    title: 'llama.cpp và Cuộc cách mạng Inference trên CPU',
    desc: 'Lý do Georgi Gerganov tạo ra llama.cpp, bài toán chạy LLM không cần GPU đắt tiền. So sánh với ONNX Runtime, TensorRT-LLM và vLLM.',
    path: '/docs/lesson_0_cpp_inference_revolution',
    category: 'Background'
  },
  {
    number: 'Bài 1',
    title: 'GGML: Thư viện Tensor bằng C không phụ thuộc',
    desc: 'Phân tích cấu trúc ggml_tensor, hệ thống 39 kiểu dữ liệu, arena memory allocation, computation graph và tối ưu SIMD cho inference.',
    path: '/docs/lesson_1_ggml_tensor_library',
    category: 'Core Theory'
  },
  {
    number: 'Bài 2',
    title: 'Định dạng GGUF: Cấu trúc nhị phân và Thiết kế',
    desc: 'Khảo sát cấu trúc file GGUF: Header, metadata key-value, tensor info, tensor data. Lập trình đọc/ghi GGUF bằng Python và C.',
    path: '/docs/lesson_2_gguf_binary_format',
    category: 'Core Theory'
  },
  {
    number: 'Bài 3',
    title: 'Giải thuật Quantization: Từ Lý thuyết đến Hiện thực',
    desc: 'Phân tích chi tiết block quantization (Q4_0-Q8_0), K-quants (super-blocks), I-quants (importance-based). Toán học quantization error và SQNR.',
    path: '/docs/lesson_3_quantization_algorithms',
    category: 'Core Theory'
  },
  {
    number: 'Bài 4',
    title: 'Kiến trúc Inference Engine của llama.cpp',
    desc: 'Đi sâu vào llama_context, KV Cache management, token sampling (top-k, top-p, temperature, min-p), batch processing và graph building.',
    path: '/docs/lesson_4_inference_engine',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 5',
    title: 'Porting Model PyTorch sang GGUF',
    desc: 'Hệ thống architecture registry, ánh xạ trọng số PyTorch sang GGUF tensors, quy trình thêm kiến trúc mô hình mới vào llama.cpp.',
    path: '/docs/lesson_5_pytorch_to_gguf',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 6',
    title: 'Hardware Backends: CPU, CUDA, Metal, Vulkan',
    desc: 'Kiến trúc backend abstraction, SIMD dispatch trên CPU, GPU offloading strategies, Apple Silicon Metal optimization, Vulkan cross-platform.',
    path: '/docs/lesson_6_hardware_backends',
    category: 'Deep Dive Code'
  },
  {
    number: 'Bài 7',
    title: 'llama.cpp Server, API và Tích hợp Hệ thống',
    desc: 'OpenAI-compatible API server, embedding generation, speculative decoding, grammar-constrained generation và các mô hình tích hợp.',
    path: '/docs/lesson_7_server_and_integration',
    category: 'Optimization'
  },
  {
    number: 'Bài 8',
    title: 'Performance Tuning và Benchmarking',
    desc: 'llama-bench, Flash Attention adaptation, context shift optimization, KV Cache quantization, chiến lược chọn quant phù hợp hardware.',
    path: '/docs/lesson_8_performance_tuning',
    category: 'Optimization'
  },
  {
    number: 'Bài 9',
    title: 'Thực hành: Tự viết GGUF Reader và Quantization',
    desc: 'Tự tay lập trình GGUF file reader bằng Python, implement Q8_0 quantization/dequantization, so sánh chất lượng với FP16 gốc.',
    path: '/docs/lesson_9_hands_on_gguf',
    category: 'Practice'
  }
];

function CategoryBadge({ category }: { category: LectureItem['category'] }) {
  const colors: Record<LectureItem['category'], { bg: string, text: string }> = {
    'Background': { bg: 'rgba(59, 130, 246, 0.15)', text: '#60a5fa' },
    'Core Theory': { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399' },
    'Deep Dive Code': { bg: 'rgba(245, 158, 11, 0.15)', text: '#fbbf24' },
    'Optimization': { bg: 'rgba(236, 72, 153, 0.15)', text: '#f472b6' },
    'Practice': { bg: 'rgba(139, 92, 246, 0.15)', text: '#a78bfa' },
  };

  const color = colors[category];

  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600, background: color.bg, color: color.text, alignSelf: 'flex-start' }}>
      {category}
    </span>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} | Deep Dive LLM Inference Architecture`}
      description="Chuỗi bài giảng phân tích chi tiết kiến trúc, giải thuật và mã nguồn của thư viện suy luận llama.cpp dành cho AI Engineers và Systems Developers.">
      <HomepageHeader />
      
      <main style={{ padding: '4rem 0', background: 'var(--ifm-background-color)' }}>
        {/* Core Pillars Section */}
        <section className="container" style={{ marginBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              🚀 Ba Đột Phá Thiết Kế Của llama.cpp
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Những cột trụ kiến trúc giúp llama.cpp trở thành thư viện inference LLM mã nguồn mở phổ biến nhất thế giới
            </p>
          </div>
          
          <div className="row">
            {CorePillars.map((item, idx) => (
              <div key={idx} className="col col--4" style={{ marginBottom: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--ifm-color-primary)', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>
                    {item.badge}
                  </span>
                  <Heading as="h3" style={{ fontSize: '1.35rem', marginBottom: '1rem', fontWeight: 600 }}>
                    {item.title}
                  </Heading>
                  <p style={{ opacity: 0.8, fontSize: '0.95rem', lineHeight: '1.6', margin: 0 }}>
                    {item.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Lectures Curriculum Section */}
        <section className="container">
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <Heading as="h2" style={{ fontSize: '2rem', fontWeight: 700 }}>
              📚 Giáo Trình Học Tập (Curriculum)
            </Heading>
            <p style={{ opacity: 0.7, maxWidth: '600px', margin: '0.5rem auto 0 auto' }}>
              Đi từ kiến thức nền tảng về inference engine, phân tích sâu cấu trúc GGUF, giải thuật quantization, kiến trúc backend đến thực hành viết GGUF Reader.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
            {Lectures.map((lecture, idx) => (
              <Link 
                to={lecture.path} 
                key={idx} 
                style={{ textDecoration: 'none', color: 'inherit' }}
              >
                <div className="glass-panel" style={{ padding: '1.5rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', cursor: 'pointer' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                      <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--ifm-color-primary)' }}>
                        {lecture.number}
                      </span>
                      <CategoryBadge category={lecture.category} />
                    </div>
                    <Heading as="h3" style={{ fontSize: '1.15rem', marginBottom: '0.75rem', fontWeight: 600, lineHeight: '1.4' }}>
                      {lecture.title}
                    </Heading>
                    <p style={{ opacity: 0.8, fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
                      {lecture.desc}
                    </p>
                  </div>
                  <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', fontSize: '0.85rem', fontWeight: 600, color: 'var(--ifm-color-primary-light)' }}>
                    Đọc bài học này <span style={{ marginLeft: '4px', transition: 'transform 0.2s ease' }} className="arrow-icon">→</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Practice Engine Banner */}
        <section className="container" style={{ marginTop: '6rem' }}>
          <div className="glass-panel" style={{ padding: '3rem', background: 'radial-gradient(circle at 90% 10%, rgba(139, 92, 246, 0.12) 0%, transparent 60%), var(--card-bg)', textAlign: 'center', borderRadius: '16px' }}>
            <Heading as="h2" style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '1rem' }}>
              💻 Thực Hành: Tự Xây Dựng GGUF Reader và Quantizer
            </Heading>
            <p style={{ maxWidth: '700px', margin: '0 auto 2rem auto', opacity: 0.8, lineHeight: '1.6' }}>
              Trong Bài 9, chúng ta sẽ tự tay triển khai bằng Python một chương trình đọc file GGUF từ đầu, implement giải thuật Q8_0 quantization/dequantization, và so sánh chất lượng trọng số với bản FP16 gốc.
            </p>
            <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <Link
                className="button button--primary button--lg"
                style={{ borderRadius: '8px', padding: '0.7rem 1.8rem', fontWeight: 600 }}
                to="/docs/lesson_9_hands_on_gguf">
                Đến Bài Học Thực Hành
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
