# State-Experiments

## State Encoder Loss Function

The pretraining of the State Encoder (SE) utilizes a decoder designed to allow the model to learn representations of the full spectrum of gene expressions. This specific decoder is **discarded** after the pretraining phase.

The input to the decoder is a concatenation of embeddings representing:
* **512 Highly Expressed Genes ($\mathcal{P}^{(i)}$):** A subset sampled from the 2,048 input genes.
* **512 Unexpressed Genes ($\mathcal{N}^{(i)}$):** Randomly sampled from genes **not** in the top 2,048 input set.
* **256 Random Genes ($\mathcal{R}$):** Sampled from the full gene set and shared across all cells in the batch.
* **Final Input Shape:** $2h + 11$ (where $h$ is the hidden dimension, plus the 10-dimensional dataset projection and 1-dimensional read depth).

Based on the paper, the total loss function for the State Encoder (SE) during pre-training is a combination of two primary objectives: a **Gene Expression Prediction** loss (which itself has two axes) and an auxiliary **Dataset Classification** loss.

The total loss is defined as:

$$
\mathcal{L} = \mathcal{L}_{expression} + \mathcal{L}_{dataset}
$$

Here is the detailed breakdown of each component:

### 1. Gene Expression Prediction Loss ($\mathcal{L}_{expression}$)
This component drives the model to learn rich cell representations by reconstructing gene expression. It is a **dual-axis** loss that simultaneously optimizes for accuracy within a single cell and consistency across the batch.

It is composed of two weighted terms:

$$
\mathcal{L}_{expression} = \lambda_{1}\mathcal{L}_{gene} + \lambda_{2}\mathcal{L}_{cell}
$$

* **Gene-Level Loss ($\mathcal{L}_{gene}$):**
    * **Goal:** Measures how well the model reconstructs the expression profile of a single cell.
    * **Function:** It computes the **L2 norm (Euclidean distance)** between the predicted and true expression vectors for the 1,280 target genes within each cell, averaged over the batch ($B$).
    * **Formula:**
    $$
    \mathcal{L}_{gene} = \frac{1}{B}\sum_{b=1}^{B}||\hat{Y}^{(b)} - Y^{(b)}||_{2}
    $$

* **Cell-Level Loss ($\mathcal{L}_{cell}$):**
    * **Goal:** Measures how well the model captures the variation of a specific gene across the population of cells in the batch.
    * **Function:** It computes the **L2 norm** between the predicted and true expression vectors for the shared set of 256 genes ($\mathcal{R}$) across all cells in the batch.
    * **Formula:**
    $$
    \mathcal{L}_{cell} = \frac{1}{|\mathcal{R}|}\sum_{r=1}^{|\mathcal{R}|}||\hat{S}'^{(r)} - S'^{(r)}||_{2}
    $$

### 2. Dataset Classification Loss ($\mathcal{L}_{dataset}$)
This auxiliary objective forces the model to disentangle technical batch effects from biological signals.

* **Goal:** Predict which experimental dataset a cell originated from using the special `[DS]` token embedding.
* **Function:** It uses a standard **Cross-Entropy Loss**.
* **Formula**:
    $$
    \mathcal{L}_{dataset} = \frac{1}{B}\sum_{b=1}^{B}\text{CrossEntropy}(\hat{d}^{(b)}, d^{(b)})
    $$
    Where $\hat{d}^{(b)}$ is the predicted dataset label and $d^{(b)}$ is the true label.

By minimizing this combined objective, the State Encoder learns embeddings that are both biologically descriptive (via expression reconstruction) and robust to technical noise (via dataset classification).

---

## Overall Loss Function: Maximum Mean Discrepancy (MMD)

The MMD is a statistical measure of distance between two probability distributions, used here to compare the population of predicted perturbed cells to the real observed cells.

### Energy Distance Kernel
The kernel used is the negative Euclidean distance of the difference between $u$ and $v$:

$$
k(u,v) = -||u-v||_{2}
$$

The kernel function $k$ is computed for every pair of predicted cell vectors and averaged (AvgP). This process is repeated for pairs within the real set (AvgR) and for cross-set pairs (between real and predicted - AvgPR).

### $\mathcal{L}_{MMD}$ Function

The squared MMD between the generated distribution $\hat{X}^{(b)}_{target}$ and the target distribution $X^{(b)}_{target}$ is given by:

$$
MMD^{2}(\hat{X}^{(b)}_{target}, X^{(b)}_{target}) = \frac{1}{S^{2}}\sum_{i=1}^{S}\sum_{j=1}^{S}[k(\hat{x}^{(i)}, \hat{x}^{(j)}) + k(x^{(i)}, x^{(j)}) - 2k(\hat{x}^{(i)}, x^{(j)})]
$$

In simpler terms:

$$
\text{Loss} = \text{AvgP} + \text{AvgR} - 2 \cdot \text{AvgPR}
$$

For a training minibatch of $B$ cell sets, we define the batch-averaged MMD loss as the average of these $MMD^{2}$ values:

$$
\mathcal{L}_{MMD}(\hat{X}_{target}, X_{target}) = \frac{1}{B}\sum_{b=1}^{B}MMD^{2}(\hat{X}^{(b)}_{target}, X^{(b)}_{target})
$$

### Final Loss Function

The final loss depends on whether the model is trained on raw expression (HVG) or embeddings (SE):

* **ST + HVG (Raw Expression):**

$$
\mathcal{L}_{total} = \mathcal{L}_{MMD}(\hat{X}_{target}, X_{target})
$$


* **SE + ST (State Embeddings):**

$$
\mathcal{L}_{total} = \mathcal{L}_{MMD}(\hat{X}_{target}^{emb}, X_{target}^{emb}) + 0.1 \cdot \mathcal{L}_{MMD}(\hat{X}_{target}, X_{target})
$$

This uses a weighted combination of MMD losses in both the embedding and gene expression spaces. The expression loss is weighted down by a factor of 0.1 to balance the two terms and avoid overwhelming the primary objective in the embedding space.
