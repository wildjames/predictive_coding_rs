# Implementation notes

I need to do only a small number of practical things - the model itself is actually fairly simple. However, I do need to be careful to be correct in my implementation.

Each node will have only a few properties
- A value, $\bf{x}_{i,t}^l$
  - randomly initialised, and will be iterated on to find the final value
- Prediction of node $(l+1)$, $\bf{u}^l_{i,l}$
$$
\bf{u}^l_{i,l} = \sum^{J_l}_{j=1}\bf{W}^{l+1}_{i,j}\phi(\bf{x}^{l+1}_{j,t})
$$
- an associated error
$$
\bf{e}^l_{i,t} = \bf{x}^l_{i,t} - \bf{u}^l_{i,t}
$$

These are not stored as individual objects, but rather as a series of arrays. Each layer will have an array of node values, weights, and errors. It will also have some activation function $\phi$ which relates it to layer $(l+1)$.

# Model running

There are two stages, first the values are found, then the model weights are adjusted.

## Value convergence

- For $0 \lt l \lt L$, i.e. the main body of the nodes,
$$
\Delta \bf{x}^l_t = \gamma \cdot (-\bf{e}^l_t + \phi'(\bf{x}^l_t) \odot (\bf{W}^l)^T \cdot \bf{e}^{l-1}_t)
$$
	- in other words, the change in value of the node at timestep $t$ is a function of its error node, plus some function of its activation function run on its own value, the weights matrix for this node, and the error nodes of the layer below this one.
- for $l = L$, if the output is unclamped (i.e. the "true" answer is not pinned), then
$$
\Delta \bf{x}^L_t = \gamma \cdot (\phi'(\bf{x^L_t}) \odot (\bf{W}^L)^T \cdot \bf{e}^{L-1}_t)
$$
	- i.e. the final later does not have a term for its own error node, and only pays attention to the error of the node in the preceding layer
- for $l=0$ in the case where the input is unclamped (i.e. where we are running inference, e.g. "generate an image of the number 4")
$$
\Delta \bf{x}^l_t = \gamma \cdot (-\bf{e}^0_t)
$$
	- i.e. the change in value of the input layer is only a function of the error node of that layer.

---

The values are initialised as random values, and time evolves according to the above until the nodes converge, or until some max timestep has passed.

## Model weight updates

Once the values of the nodes have converged (or near enough), then we perform a weight update step. This is not the same as a typical back propagation, but does still minimse the same energy function as before:
$$
\Delta\bf{W}^{l+1} = -\alpha (\partial\varepsilon_T / \partial \bf{W}^{l+1}) = \alpha (\bf{e}^l_T \cdot \phi(\bf{x}_T^{l+1})^T)
$$
where $\alpha$ is the learning rate for the synapses. Note that the subscript $T$ denotes that the values have converged. So, the weight updates of a layer $(l+1)$ are dependant only on the error of the previous layer, and the result of passing the values of $(l+1)$ through its activation function $\phi$.
