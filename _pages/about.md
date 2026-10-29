---
layout: about
title: about
permalink: /
subtitle: Research Scientist at Devotion AI Labs

profile:
  align: right
  image: prof_pic.png
  image_circular: false
  more_info: >

#news: true
latest_posts: false
social: true
---

I'm a Research Scientist at Devotion AI Labs. Before that, I spent several years at Cambridge, completing my PhD with Prof. Carola-Bibiane Schönlieb in the Department of Applied Mathematics and Theoretical Physics (DAMTP), then working with Prof. Mark Girolami as a Postdoctoral Research Associate in the Department of Engineering.

My current research focuses on **diffusion language models (DLMs)** and closing the capability gap with autoregressive language models, commonly known as LLMs. In [_CoBit_](https://arxiv.org/abs/2605.07013) (Continuous Bitstream Diffusion), we represent text as bitstreams and model it with continuous diffusion. At the time of our arXiv submission, our GenPPL–entropy and GSM8K results suggested an advance in the state of the art among continuous DLMs on those evaluations. The field has moved on rapidly since, so this should be read as a historical result.

The collective progress of the DLM research community makes me optimistic that the capability gap could close sooner than we expect. Matching strong autoregressive models across a broad range of tasks remains an open challenge, but the momentum is exciting. I look forward to sharing new work soon and contributing to that progress.

As the capability gap narrows, I expect speed and inference cost to become increasingly decisive. I'm excited by the prospect of DLMs competing as standalone models in production, and by their potential as parallel draft models within speculative decoding, proposing tokens for an autoregressive model to verify. Both directions offer opportunities for faster generation, provided the gains hold up in practical workloads without compromising quality.

My interest in DLMs also connects to my earlier work on the geometry of continuous data. There, I have studied how trained diffusion models and normalising flows can define Riemannian metrics that reveal intrinsic structure and give meaningful notions of distance and interpolation. I'm particularly interested in how this perspective might extend to discrete distributions, including text and DNA sequences. What notions of geometry are useful for these data, and how can they guide the choice of representations and the design of more efficient diffusion models?

**At NeurIPS 2026:** I will be presenting CoBit in Sydney and would be happy to meet others working on DLMs. If you will be there, feel free to get in touch.

### Research interests

- **Language and multimodal generation:** designing unified generative models that capture relationships across modalities and support both joint and conditional generation. Our [multimodal CoBit work](https://arxiv.org/abs/2610.14349) explores one approach.
- **Fast, capable diffusion models:** accelerating inference through caching, parallel generation, and distillation, while understanding the trade-offs between speed and quality.
- **Information theory and optimal sampling:** understanding what makes denoising difficult and designing sampling algorithms with guarantees on accuracy and computational cost.
- **Learning the geometry of data:** constructing Riemannian metrics from trained generative models and extending geometric methods from continuous data to text and biological sequences.

If any of these topics interest you, feel free to reach out for a chat at [g.batz97@gmail.com](mailto:g.batz97@gmail.com) or [Georgios@devotionlabs.ai](mailto:Georgios@devotionlabs.ai).

### Publications

<div class="publication-section">
  <div class="publication-entry">
    <div class="publication-label">NeurIPS 2026</div>
    <div class="publication-content">
      <div class="publication-title">CoBit: Language Modeling with Bitstream Diffusion</div>
      <div class="publication-authors"><strong>Georgios Batzolis</strong>, Mark Girolami, Luca Ambrogioni</div>
      <div class="publication-status">Accepted at NeurIPS 2026; also appeared at ICML SPIGM 2026 and the <a href="https://pengzhangzhi.github.io/NonAR-LM/">COLM 2026 workshop on Non-Autoregressive Language Models for Fast &amp; Flexible Text Generation</a>.</div>
      <br>
      <a href="https://arxiv.org/pdf/2605.07013">[PDF]</a>
      <a href="https://github.com/GBATZOLIS/BitstreamDiffusion">[Code]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">NeurIPS 2026</div>
    <div class="publication-content">
      <div class="publication-title">Compatible Likelihoods for Flow Matching on Manifolds</div>
      <div class="publication-authors">Lucas Ng, <strong>Georgios Batzolis</strong>, Mark Girolami</div>
      <div class="publication-status">Accepted at NeurIPS 2026.</div>
      <br>
      <a href="https://arxiv.org/abs/2610.31981">[arXiv]</a>
      <a href="https://arxiv.org/pdf/2610.31981">[PDF]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">ICML 2025</div>
    <div class="publication-content">
      <div class="publication-title"><a href="/projects/riemanniangeometry">Score-based Pullback Riemannian Geometry</a></div>
      <div class="publication-authors">Willem Diepeveen*, <strong>Georgios Batzolis</strong>*, Zakhar Shumaylov, Carola-Bibiane Schönlieb</div>
      <div class="publication-status">ICML 2025</div>
      <br>
      <a href="https://openreview.net/pdf?id=AJN5btaqNk">[Published Version]</a>
      <a href="https://github.com/GBATZOLIS/Score-Based-Pullback-Riemannian-Geometry">[Code]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">ICML 2024</div>
    <div class="publication-content">
      <div class="publication-title"><a href="/projects/id_diff">Diffusion Models Encode the Intrinsic Dimension of Data Manifolds</a></div>
      <div class="publication-authors">Jan Stanczuk*, <strong>Georgios Batzolis</strong>*, Teo Deveney, Carola-Bibiane Schönlieb</div>
      <div class="publication-status">ICML 2024</div>
      <br>
      <a href="https://openreview.net/pdf?id=a0XiA6v256">[Published Version]</a>
      <a href="https://arxiv.org/abs/2212.12611">[arXiv]</a>
      <a href="https://github.com/GBATZOLIS/ID-diff">[Official Code]</a>
      <a href="https://github.com/GBATZOLIS/ID-diff-clean">[Demo/Clean Code]</a>
      <a href="https://gbatzolis.github.io/ID-diff/">[Official Webpage]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">FoDS 2024</div>
    <div class="publication-content">
      <div class="publication-title">CAFLOW: conditional autoregressive flows</div>
      <div class="publication-authors"><strong>Georgios Batzolis</strong>, Marcello Carioni, Christian Etmann, Soroosh Afyouni, Zoe Kourtzi, Carola-Bibiane Schönlieb</div>
      <div class="publication-status">Foundations of Data Science, AIMS, 2024</div><br>
      DOI: 10.3934/fods.2024028
      <br>
      <a href="https://arxiv.org/abs/2106.02531">[arXiv]</a>
      <a href="https://github.com/GBATZOLIS/CAFLOW">[Official Code]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">AAAI 2022</div>
    <div class="publication-content">
      <div class="publication-title">How to distribute data across tasks for meta-learning?</div>
      <div class="publication-authors">Alexandru Cioba, Michael Bromberg, Qian Wang, Ritwik Niyogi, <strong>Georgios Batzolis</strong>, Jezabel Garcia, Da-shan Shiu, Alberto Bernacchia</div>
      <div class="publication-status">AAAI Conference on Artificial Intelligence, 2022</div>
      <br>
      <a href="https://arxiv.org/abs/2103.08463">[PDF]</a>
    </div>
  </div>
</div>

### arXiv Preprints

<div class="publication-section">
  <div class="publication-entry">
    <div class="publication-label">arXiv 2026</div>
    <div class="publication-content">
      <div class="publication-title">Multimodal Co-Generation With Continuous Bitstream Diffusion</div>
      <div class="publication-authors"><strong>Georgios Batzolis</strong>, Georgios Kontos, Bruno Trentini, Pranam Chatterjee, Micha Livne, Mark Girolami, Luca Ambrogioni</div>
      <div class="publication-status">Submitted to ICLR 2027.</div>
      <br>
      <a href="https://arxiv.org/abs/2610.14349">[arXiv]</a>
      <a href="https://arxiv.org/pdf/2610.14349">[PDF]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">arXiv 2026</div>
    <div class="publication-content">
      <div class="publication-title">Beyond Unimodal Bases: Pullback Geometry for Multimodal Data</div>
      <div class="publication-authors">Honglei Brinkmann, Lucas Ng, <strong>Georgios Batzolis</strong>, Mark Girolami, Carola-Bibiane Schönlieb, Willem Diepeveen</div>
      <div class="publication-status">Submitted to ICLR 2027.</div>
      <br>
      <a href="https://arxiv.org/abs/2610.00708">[arXiv]</a>
      <a href="https://arxiv.org/pdf/2610.00708">[PDF]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">arXiv 2026</div>
    <div class="publication-content">
      <div class="publication-title">From Atoms to Entropy: Optimal Noise Allocation for Diffusion Training in the Convex Regime</div>
      <div class="publication-authors">Luca Ambrogioni, Giulio Franzese, Alberto Foresti, Gabriel Raya, Bac Nguyen, <strong>Georgios Batzolis</strong>, Yuhta Takida, Naoki Murata, Chieh-Hsin Lai, Yuki Mitsufuji</div>
      <div class="publication-status">Submitted to AISTATS.</div>
      <br>
      <a href="https://arxiv.org/abs/2607.20540">[arXiv]</a>
      <a href="https://arxiv.org/pdf/2607.20540">[PDF]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">arXiv 2026</div>
    <div class="publication-content">
      <div class="publication-title">Noise Scheduling as Information-Guided Allocation in Diffusion Training</div>
      <div class="publication-authors">Gabriel Raya, Bac Nguyen, <strong>Georgios Batzolis</strong>, Yuhta Takida, Dejan Stancevic, Naoki Murata, Chieh-Hsin Lai, Yuki Mitsufuji, Luca Ambrogioni</div>
      <div class="publication-status">Under review; also appeared at <a href="https://openreview.net/forum?id=VnnK7CYAx2">ICML SPIGM 2026</a>.</div>
      <br>
      <a href="https://arxiv.org/abs/2602.18647">[arXiv]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">arXiv 2023</div>
    <div class="publication-content">
      <div class="publication-title"><a href="/projects/scorevae">Latent Space Extraction from Pre-trained Diffusion Models</a></div>
      <div class="publication-authors"><strong>Georgios Batzolis</strong>*, Jan Stanczuk*, Carola-Bibiane Schönlieb</div>
      <div class="publication-status">arXiv preprint arXiv:2304.12141</div>
      <br>
      <a href="https://arxiv.org/abs/2304.12141">[PDF]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">arXiv 2022</div>
    <div class="publication-content">
      <div class="publication-title">Non-uniform diffusion models</div>
      <div class="publication-authors"><strong>Georgios Batzolis</strong>, Jan Stanczuk, Carola-Bibiane Schönlieb</div>
      <div class="publication-status">arXiv preprint arXiv:2207.09786</div>
      <br>
      <a href="https://arxiv.org/abs/2207.09786">[PDF]</a>
    </div>
  </div>

  <div class="publication-entry">
    <div class="publication-label">arXiv 2021</div>
    <div class="publication-content">
      <div class="publication-title">Conditional image generation with score-based diffusion models</div>
      <div class="publication-authors"><strong>Georgios Batzolis</strong>, Jan Stanczuk, Carola-Bibiane Schönlieb, Christian Etmann</div>
      <div class="publication-status">arXiv preprint arXiv:2111.13606</div>
      <br>
      <a href="https://arxiv.org/abs/2111.13606">[PDF]</a>
    </div>
  </div>
</div>

<p class="publication-note">* Equal contribution.</p>

### Contact

Email: [g.batz97@gmail.com](mailto:g.batz97@gmail.com) or [Georgios@devotionlabs.ai](mailto:Georgios@devotionlabs.ai). You can also find me on [Google Scholar](https://scholar.google.com/citations?user=IkmiaicAAAAJ) or [LinkedIn](https://www.linkedin.com/in/georgios-batzolis-92577b128/).
