---
layout: page
title: "Diffusion Models Encode the Intrinsic Dimension of Data Manifolds"
permalink: /projects/id_diff
nav: false
math: true
---

<style>
    .algorithm {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        font-family: Arial, sans-serif;
        max-width: 600px;
        margin-left: 0;
    }
    .algorithm-title {
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .input-output {
        font-weight: bold;
        margin-top: 10px;
    }
    .algorithm ol, .algorithm ul {
        margin-left: 20px;
        padding-left: 0;
    }
    .algorithm li {
        margin-bottom: 8px;
    }
    .algorithm .step {
        margin-left: 20px;
    }
</style>

<div style="display: flex; justify-content: center;">
  <img src="/assets/img/id_diff/score_field.png" alt="Score Field" style="flex: 1; max-width: 400px; object-fit: contain;">
  <img src="/assets/img/id_diff/drawing.png" alt="Drawing" style="flex: 1; max-width: 400px; object-fit: contain;">
</div>

## Overview

In this work, we present theoretical and experimental evidence that diffusion models—widely used in generative modeling—effectively capture the **intrinsic dimension** of data. We introduce a novel method for estimating this intrinsic dimension directly from a trained diffusion model.

---

## Problem Setup

We consider the following setup:

<div class="mathjax_process">
<ol>
  <li>There is a <strong>data manifold</strong> \( \mathcal{M} \) of <strong>intrinsic dimension</strong> \( k \), embedded in an ambient Euclidean space \( \mathbb{R}^d \), where \( d \gg k \).</li>
  <li>There is a probability distribution \( p \) that is highly concentrated around \( \mathcal{M} \).</li>
  <li>We are given a finite sample of data \( \{ x_i \}_{i=1}^n \subseteq \mathbb{R}^d \) generated from \( p \).</li>
</ol>

Diffusion models are generative models designed to learn \( p \), but they don't explicitly find \( k \). We show how one can extract \( k \) by training a diffusion model on the data.
</div>

---

## Diffusion Models and the Score Function

We consider an Itô diffusion:

<div class="mathjax_process">
\[
dx = f(x, t)\, dt + g(t)\, dW
\]
</div>

The time-reversed stochastic differential equation (SDE) is:

<div class="mathjax_process">
\[
dx = \left[ f(x, t) - g(t)^2 \nabla_x \ln p_t(x) \right] dt + g(t)\, dW
\]
</div>

Diffusion models \( s_\theta(x, t) \) are trained to approximate the score function \( \nabla_x \ln p_t(x) \), which represents the gradient of the log-density of the noisy data distribution at time \( t \).

<div style="text-align: center;">
  <img src="/assets/img/id_diff/forward_reverse.jpg" alt="Forward and Reverse SDEs" style="width: 100%; max-width: 600px; height: auto;">
</div>

*Figure: Forward and reverse SDEs*

---

## Key Observation: Score Field Perpendicular to the Manifold

<div class="mathjax_process">
For small diffusion times \( \varepsilon \), the score function \( \nabla_x \ln p_{\varepsilon}(x) \) becomes approximately perpendicular to the manifold \( \mathcal{M} \).

To formalize this, let \( \pi(x) \) be the projection of \( x \) onto \( \mathcal{M} \), \( \mathcal{N}_{\pi(x)}\mathcal{M} \) the normal space, and \( \mathcal{T}_{\pi(x)}\mathcal{M} \) the tangent space at \( \pi(x) \).

The score vector \( \nabla_x \ln p_{\varepsilon}(x) \) primarily lies within the normal space \( \mathcal{N}_{\pi(x)}\mathcal{M} \). Mathematically:

\[
\left\| \text{Proj}_{\mathcal{N}_{\pi(x)}\mathcal{M}} \left[ \nabla_x \ln p_{\varepsilon}(x) \right] \right\| \gg \left\| \text{Proj}_{\mathcal{T}_{\pi(x)}\mathcal{M}} \left[ \nabla_x \ln p_{\varepsilon}(x) \right] \right\|.
\]

This phenomenon occurs because, at small \( \varepsilon \), \( p_{\varepsilon}(x) \) is sharply concentrated around \( \mathcal{M} \).
</div>

<div style="display: flex; justify-content: center;">
  <img src="/assets/img/id_diff/score_field.png" alt="Score Field" style="flex: 1; max-width: 400px; object-fit: contain;">
  <img src="/assets/img/id_diff/drawing.png" alt="Drawing" style="flex: 1; max-width: 400px; object-fit: contain;">
</div>

---

## Intrinsic Dimension Estimation Method

This observation suggests that by sampling sufficiently many perturbed points around a datapoint \( x_0 \in \mathcal{M} \), the score vectors at these points will span the normal space \( \mathcal{N}_{x_0}\mathcal{M} \). The following algorithm estimates the intrinsic dimension from a trained diffusion model.

<div class="algorithm">
    <div class="algorithm-title">Algorithm: Intrinsic Dimension Estimation</div>

    <div class="input-output">Input:</div>
    <ul>
        <li>\( s_\theta \): trained diffusion model (score function)</li>
        <li>\( t_0 \): small sampling time</li>
        <li>\( K \): number of perturbed points</li>
    </ul>

    <div class="input-output">Algorithm:</div>
    <ol>
        <li>Sample \( \mathbf{x}_0 \) from the dataset \( p_0(\mathbf{x}) \).</li>
        <li>Set \( d = \text{dim}(\mathbf{x}_0) \).</li>
        <li>Initialize an empty matrix \( S \).</li>
        <li>For \( i = 1, \dots, K \):
            <ul>
                <li>Sample \( \mathbf{x}_{t_0}^{(i)} \sim \mathcal{N}(\mathbf{x}_0, \sigma_{t_0}^2 \mathbf{I}) \).</li>
                <li>Append \( s_\theta(\mathbf{x}_{t_0}^{(i)}, t_0) \) as a new column to \( S \).</li>
            </ul>
        </li>
        <li>Perform SVD on \( S \) to obtain singular values \( (s_i)_{i=1}^d \).</li>
        <li>Estimate \( \hat{k}(\mathbf{x}_0) = d - \arg\max_{i=1,\dots,d-1} (s_i - s_{i+1}) \).</li>
    </ol>

    <div class="input-output">Output:</div>
    <ul>
        <li>Estimated intrinsic dimension \( \hat{k}(\mathbf{x}_0) \).</li>
    </ul>
</div>

---

## Theoretical Results

Using tubular neighborhood and Morse theory, we rigorously proved the following theorem, confirming our intuition. Full details of the proof are provided in Appendix D of the paper.

### Theorem

<div class="mathjax_process">
The ratio of the projection of the score \( \nabla_{\mathbf{x}} \ln p_t(\mathbf{x}) \) onto the tangent space \( T_{\pi(\mathbf{x})}\mathcal{M} \) to the normal space \( \mathcal{N}_{\pi(\mathbf{x})}\mathcal{M} \) approaches zero as \( t \to 0 \), i.e.,

\[
\frac{\|\mathbf{T} \nabla_{\mathbf{x}} \ln p_t(\mathbf{x})\|}{\|\mathbf{N} \nabla_{\mathbf{x}} \ln p_t(\mathbf{x})\|} \to 0, \quad \text{as } t \to 0.
\]

Thus, for sufficiently small \( t \), the score \( \nabla_{\mathbf{x}} \ln p_t(\mathbf{x}) \) is effectively contained in \( \mathcal{N}_{\pi(\mathbf{x})}\mathcal{M} \).
</div>

---

## Experiments

We validated our method on various datasets, comparing it with traditional intrinsic dimension estimation techniques.

### Synthetic Manifolds

We generated \( k \)-dimensional spheres embedded in \( \mathbb{R}^{100} \) using random isometric embeddings.

-

 **10-Sphere** (\( k = 10 \)): Estimated \( \hat{k} = 11 \)
- **50-Sphere** (\( k = 50 \)): Estimated \( \hat{k} = 51 \)

<div style="text-align: center;">
  <img src="/assets/img/id_diff/10_sphere.png" alt="10-Sphere" style="width: 49%; max-width: 400px;">
  <img src="/assets/img/id_diff/50_sphere.png" alt="50-Sphere" style="width: 49%; max-width: 400px;">
</div>

#### Image Manifolds: Square and Gaussian Blobs

<div style="display: flex; justify-content: center; gap: 10px; text-align: center;">
  <figure style="width: 32%; max-width: 250px;">
    <img src="/assets/img/id_diff/figures/squares/square_image_10D.png" alt="Square 10D" style="width: 100%;">
    <figcaption>k=10</figcaption>
  </figure>
  <figure style="width: 32%; max-width: 250px;">
    <img src="/assets/img/id_diff/figures/squares/square_image_20D.png" alt="Square 20D" style="width: 100%;">
    <figcaption>k=20</figcaption>
  </figure>
  <figure style="width: 32%; max-width: 250px;">
    <img src="/assets/img/id_diff/figures/squares/square_image_100D.png" alt="Square 100D" style="width: 100%;">
    <figcaption>k=100</figcaption>
  </figure>
</div>

---

## Limitations

- **Approximation Error**: Caused by imperfect score approximation \( s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \ln p_t(\mathbf{x}) \).
- **Geometric Error**: Arises when \( t \) isn't sufficiently small, causing:
    - Increased tangential components of the score vector.
    - Differences in normal spaces across sampled points due to curvature.

---

## Conclusions

- Our estimator accurately captures intrinsic dimensions, outperforming traditional methods.
- The inductive biases in the neural network estimating the score function are key.
- We can potentially extract other properties of the data manifold, such as curvature, from a trained diffusion model.
