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
        margin-left: 0; /* Align to the left */
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

In this work, we theoretically and experimentally prove that diffusion models, extensively used in generative modeling, capture the **intrinsic dimension** of the data. We introduce a novel method for estimating this intrinsic dimension directly from a trained diffusion model.

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

Diffusion models \( s_\theta(x, t) \) are essentially trained to approximate the score function \( \nabla_x \ln p_t(x) \), which represents the gradient of the log-density of the noisy data distribution at time \( t \). Even if they are trained to predict the noise \(n\) or the unperturbed image \(x_0\), their output can be trivially rescaled and shifted to approximate the score function.
</div>

<div style="text-align: center;">
  <img src="/assets/img/id_diff/forward_reverse.jpg" alt="Forward and Reverse SDEs" style="width: 100%; max-width: 600px; height: auto;">
</div>

*Figure: Forward and reverse SDEs*

---

## Key Observation: Score Field Perpendicular to the Manifold
<div class="mathjax_process">
For small diffusion times \( \varepsilon \), the score function \( \nabla_x \ln p_{\varepsilon}(x) \) becomes approximately perpendicular to the manifold \( \mathcal{M} \). <br><br>

<p> To formalize this observation, let \( \pi(x) \) denote the projection of the point \( x \) onto \( \mathcal{M} \), \( \mathcal{N}_{\pi(x)}\mathcal{M} \) denote the normal space and \( \mathcal{T}_{\pi(x)}\mathcal{M} \) denote the tangent space of \( \mathcal{M} \) at \( \pi(x) \).

The score vector \( \nabla_x \ln p_{\varepsilon}(x) \) predominantly lies within the normal space \( \mathcal{N}_{\pi(x)}\mathcal{M} \). Mathematically, this means that its projection onto the normal space is significantly larger than its projection onto the tangent space:

\[
\left\| \text{Proj}_{\mathcal{N}_{\pi(x)}\mathcal{M}} \left[ \nabla_x \ln p_{\varepsilon}(x) \right] \right\| \gg \left\| \text{Proj}_{\mathcal{T}_{\pi(x)}\mathcal{M}} \left[ \nabla_x \ln p_{\varepsilon}(x) \right] \right\|.
\]

This phenomenon occurs because, at small \( \varepsilon \), the probability density \( p_{\varepsilon}(x) \) is sharply concentrated around \( \mathcal{M} \). The variations in \( p_{\varepsilon}(x) \) are much more pronounced in the normal direction than in the tangent direction, causing the gradient \( \nabla_x \ln p_{\varepsilon}(x) \) to point primarily towards \( \mathcal{M} \) along the normal.</p>
</div>

<div style="display: flex; justify-content: center;">
  <img src="/assets/img/id_diff/score_field.png" alt="Score Field" style="flex: 1; max-width: 400px; object-fit: contain;">
  <img src="/assets/img/id_diff/drawing.png" alt="Drawing" style="flex: 1; max-width: 400px; object-fit: contain;">
</div>

---

## Intrinsic Dimension Estimation Method
<div class="mathjax_process">
This key observation suggests that by sampling sufficiently many perturbed points around a datapoint \( x_0 \in \mathcal{M} \), the score vectors at these points will span the normal space \( \mathcal{N}_{x_0}\mathcal{M} \). This means that if we collect the score vectors in a matrix and perform svd on this matrix, we should see a spectrum drop exactly at the dimension of the normal space \(\mathcal{N}_{x_0}\mathcal{M}\). This insight forms the basis of the following straightforward algorithm for extracting the intrinsic dimension from a trained diffusion model.


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
        <li>Perform Singular Value Decomposition (SVD) on \( S \) to obtain singular values \( (s_i)_{i=1}^d \).</li>
        <li>Estimate \( \hat{k}(\mathbf{x}_0) = d - \arg\max_{i=1,\dots,d-1} (s_i - s_{i+1}) \).</li>
    </ol>

    <div class="input-output">Output:</div>
    <ul>
        <li>Estimated intrinsic dimension \( \hat{k}(\mathbf{x}_0) \).</li>
    </ul>
</div>
</div>

---

## Theoretical Results

Using the notion of tubular neighbourhood and Morse theory,  we rigorously proved the following theorem, confirming our initial intuition. Full details of the proof are provided in Appendix D of the paper.

### Theorem

<div class="mathjax_process">
The ratio of the projection of the score \( \nabla_{\mathbf{x}} \ln p_t(\mathbf{x}) \) onto the tangent space of the data manifold \( T_{\pi(\mathbf{x})}\mathcal{M} \) to its projection onto the normal space \( \mathcal{N}_{\pi(\mathbf{x})}\mathcal{M} \) approaches zero as \( t \) approaches zero, i.e.,
\[
\frac{\|\mathbf{T} \nabla_{\mathbf{x}} \ln p_t(\mathbf{x})\|}{\|\mathbf{N} \nabla_{\mathbf{x}} \ln p_t(\mathbf{x})\|} \to 0, \quad \text{as } t \to 0.
\]
Here, \( \mathbf{N} \) and \( \mathbf{T} \) are projection matrices onto \( \mathcal{N}_{\pi(\mathbf{x})}\mathcal{M} \) and \( T_{\pi(\mathbf{x})}\mathcal{M} \), respectively. Thus, for sufficiently small \( t \), the score \( \nabla_{\mathbf{x}} \ln p_t(\mathbf{x}) \) is (effectively) contained in the normal space \( \mathcal{N}_{\pi(\mathbf{x})}\mathcal{M} \).
</div>

---

## Experiments

We validated our method on various datasets, both synthetic and real-world, comparing it with traditional intrinsic dimension estimation techniques.

### Synthetic Manifolds
<div class="mathjax_process">
<h4>\( k \)-Spheres in 100 Dimensions</h4>


<p>We generate \( k \)-dimensional spheres embedded in \( \mathbb{R}^{100} \) using random isometric embeddings.</p>
</div>

<ul>
  <li>10-Sphere (\( k = 10 \)): Estimated \( \hat{k} = 11 \)</li>
  <li>50-Sphere (\( k = 50 \)): Estimated \( \hat{k} = 51 \)</li>
</ul>

<div style="text-align: center;">
  <img src="/assets/img/id_diff/10_sphere.png" alt="10-Sphere" style="width: 49%; max-width: 400px;">
  <img src="/assets/img/id_diff/50_sphere.png" alt="50-Sphere" style="width: 49%; max-width: 400px;">
</div>

#### Mammoth Manifold

<div class="mathjax_process">
A 2D manifold embedded in 100 dimensions. Estimated: \( \hat{k} = 2 \)
</div>

<div style="text-align: center;">
  <img src="/assets/img/id_diff/mammoth.png" alt="Mammoth Manifold" style="width: 39%; max-width: 300px;">
  <img src="/assets/img/id_diff/mammoth_spectrum.png" alt="Mammoth Spectrum" style="width: 59%; max-width: 500px;">
</div>

#### Spaghetti Line in 100 Dimensions

<div class="mathjax_process">
A 1D curve embedded in \( \mathbb{R}^{100} \). Estimated: \( \hat{k} = 1 \)
</div>



<div style="text-align: center;">
  <img src="/assets/img/id_diff/spaghetti.png" alt="Spaghetti Line" style="width: 39%; max-width: 300px;">
  <img src="/assets/img/id_diff/line.png" alt="Line Spectrum" style="width: 59%; max-width: 500px;">
</div>

#### Union of Manifolds: 10-Sphere and 30-Sphere

We analyze a dataset composed of a union of two spheres.

<div style="text-align: center;">
  <img src="/assets/img/id_diff/union_spectrum.png" alt="Union Spectrum" style="width: 100%; max-width: 600px;">
</div>

The estimated dimensions vary locally, reflecting the manifold's structure.

<div style="text-align: center;">
  <img src="/assets/img/id_diff/union_dims.png" alt="Union Dimensions" style="width: 100%; max-width: 600px;">
</div>

### Image Manifolds

#### Square Manifold

Samples from the Square Manifold for different dimensions:

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


Spectra of singular values:

<div style="text-align: center;">
  <img src="/assets/img/id_diff/figures/squares/squares_spectrum.png" alt="Square Spectrum" style="width: 48%; max-width: 400px;">
  <img src="/assets/img/id_diff/figures/squares/squares_spectrum_adhoc.png" alt="Square Spectrum Adhoc" style="width: 48%; max-width: 400px;">
</div>

#### Gaussian Blobs Manifold

Samples from the Gaussian Blobs Manifold:

<div style="display: flex; justify-content: center; gap: 10px; text-align: center;">
  <figure style="width: 32%; max-width: 250px;">
    <img src="/assets/img/id_diff/figures/blobs/gaussian_image_10D.png" alt="Gaussian 10D" style="width: 100%;">
    <figcaption>k=10</figcaption>
  </figure>
  <figure style="width: 32%; max-width: 250px;">
    <img src="/assets/img/id_diff/figures/blobs/gaussian_image_20D.png" alt="Gaussian 20D" style="width: 100%;">
    <figcaption>k=20</figcaption>
  </figure>
  <figure style="width: 32%; max-width: 250px;">
    <img src="/assets/img/id_diff/figures/blobs/gaussian_image_100D.png" alt="Gaussian 100D" style="width: 100%;">
    <figcaption>k=100</figcaption>
  </figure>
</div>


Spectra of singular values:

<div style="text-align: center;">
  <img src="/assets/img/id_diff/figures/blobs/gaussians_spectrum.png" alt="Gaussian Spectrum" style="width: 48%; max-width: 400px;">
  <img src="/assets/img/id_diff/figures/blobs/gaussians_spectrum_adhoc.png" alt="Gaussian Spectrum Adhoc" style="width: 48%; max-width: 400px;">
</div>

### Real-World Data: MNIST

We applied our method to the MNIST handwritten digits dataset, analyzing each digit class separately.

<div style="display: flex; flex-wrap: wrap; align-items: flex-start;">
  <div style="width: 60%; max-width: 600px;">
    <img src="/assets/img/id_diff/figures/MNIST/mnist_spectrum.png" alt="MNIST Score Spectra" style="width: 100%; height: auto;">
    <p style="text-align: center;">Figure: MNIST Score Spectra</p>
  </div>
  <div style="width: 38%; max-width: 380px; margin-left: 2%;">
    <img src="/assets/img/id_diff/figures/MNIST/mnist_autoencoder.png" alt="Autoencoder Validation" style="width: 100%; height: auto;">
    <p style="text-align: center;">Figure: Autoencoder Validation</p>
  </div>
</div>

Estimated intrinsic dimensions for each digit:

<table style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <th>Digit</th>
    <th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th>
  </tr>
  <tr>
    <td><strong>Estimated \( \hat{k} \)</strong></td>
    <td>113</td><td>66</td><td>131</td><td>120</td><td>107</td><td>129</td><td>126</td><td>100</td><td>148</td><td>152</td>
  </tr>
</table>

### Summary of Experimental Results

We compared our method with other intrinsic dimension estimation techniques.

<table style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <th>Dataset</th>
    <th>Ground Truth</th>
    <th>Ours</th>
    <th>MLE (m=5)</th>
    <th>MLE (m=20)</th>
    <th>Local PCA</th>
    <th>PPCA</th>
  </tr>
  <tr>
    <td colspan="7"><strong>Euclidean Data Manifolds</strong></td>
  </tr>
  <tr>
    <td>10-Sphere</td>
    <td>10</td>
    <td>11</td>
    <td>9.61</td>
    <td>9.46</td>
    <td>11</td>
    <td>11</td>
  </tr>
  <tr>
    <td>50-Sphere</td>
    <td>50</td>
    <td>51</td>
    <td>35.52</td>
    <td>34.04</td>
    <td>51</td>
    <td>51</td>
  </tr>
  <tr>
    <td>Spaghetti Line</td>
    <td>1</td>
    <td>1</td>
    <td>1.01</td>
    <td>1.00</td>
    <td>32</td>
    <td>98</td>
  </tr>
  <tr>
    <td colspan="7"><strong>Image Manifolds</strong></td>
  </tr>
  <tr>
    <td>Squares (k=10)</td>
    <td>10</td>
    <td>11</td>
    <td>8.48</td>
    <td>8.17</td>
    <td>10</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Squares (k=20)</td>
    <td>20</td>
    <td>22</td>
    <td>14.96</td>
    <td>14.36</td>
    <td>20</td>
    <td>20</td>
  </tr>
  <tr>
    <td>Squares (k=100)</td>
    <td>100</td>
    <td>100</td>
    <td>37.69</td>
    <td>34.42</td>
    <td>78</td>
    <td>99</td>
  </tr>
  <tr>
    <td>Gaussian Blobs (k=10)</td>
    <td>10</td>
    <td>12</td>
    <td>8.88</td>
    <td>8.67</td>
    <td>10</td>
    <td>136</td>
  </tr>
  <tr>
    <td>Gaussian Blobs (k=20)</td>
    <td>20</td>
    <td>21</td>
    <td>16.34</td>
    <td>15.75</td>
    <td>20</td>
    <td>264</td>
  </tr>
  <tr>
    <td>Gaussian Blobs (k=100)</td>
    <td>100</td>
    <td>98</td>
    <td>39.66</td>
    <td>35.31</td>
    <td>18</td>
    <td>985</td>
  </tr>
  <tr>
    <td colspan="7"><strong>MNIST</strong></td>
  </tr>
  <tr>
    <td>All Digits</td>
    <td>N/A</td>
    <td>152</td>
    <td>14.12</td>
    <td>13.27</td>
    <td>38</td>
    <td>706</td>
  </tr>
</table>

---

## Limitations
<div class="mathjax_process">
<ul>
  <li><strong>Approximation Error</strong>: Caused by imperfect score approximation \( s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \ln p_t(\mathbf{x}) \).</li>
  <li><strong>Geometric Error</strong>: Arises when \( t \) isn't sufficiently small, leading to:
    <ul>
      <li>Increased tangential components of the score vector.</li>
      <li>Differences in normal spaces across sampled points due to manifold curvature.</li>
    </ul>
  </li>
</ul>
</div>

---

## Conclusions

- Our estimator offers accurate intrinsic dimension estimates even for high-dimensional manifolds, indicating superior statistical efficiency compared to traditional methods.
- This improvement is credited to the inductive biases of the neural network estimating the score function, the critical quantity for intrinsic dimension estimation.
- Our theoretical results show that the diffusion model approximates the normal bundle of the manifold, providing more information than just the intrinsic dimension.
- We can potentially use a trained diffusion model to extract other important properties of the data manifold, such as curvature.
