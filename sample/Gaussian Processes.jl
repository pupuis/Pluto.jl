### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 1186151e-f6cf-11ea-1c3c-13083a1768e8
begin
	using Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 1626d748-f6cf-11ea-16cf-ad8d3458005c
begin
	Pkg.add(["Plots", "Distributions", "Distances", "PlutoUI"])
	using Plots
	using Distributions
	using Distances
	using LinearAlgebra
	using PlutoUI
end

# ╔═╡ e125117a-f75c-11ea-316c-87d98ff4fd33
md"""
# Gaussian Processes with Pluto
In this notebook we are going to explore the world of Gaussian processes (GPs), an interesting machine learning approach that can be used for e.g. curve-fitting.
We start by creating a clean environment and adding some packages. 
"""

# ╔═╡ de5dddac-f75d-11ea-0943-27cd853abb42
md"""
You may find that we are not going to use any dedicated machine learning libraries for Julia in this notebook. This is because we can construct our problem setting as well as our model ourselves in just a few lines of code. Most theoretical background in this notebook is based on [Gaussian Processes for Machine Learning (2006)](http://www.gaussianprocess.org/gpml/) by Rasmussen and Williams. There exists a great deal of interesting guides to GPs on the internet and I will try to link as many of them as possible in the end of this notebook.
Now let us get going - with nothing but an old radio device and a piece of paper...
"""

# ╔═╡ a302f556-f6d0-11ea-09b7-1b41dd4c9d1e
md"""
## A function-fitting Problem

Assume you register a noisy signal on an old radio device and transcribe it to a piece of paper, something like this:
"""

# ╔═╡ ee1dc330-f6d1-11ea-1164-cbef833a75a2
md"""
It looks quite messy but you strongly believe that behind all that noise there might be a rather interesting signal. And, as we are going to find out, you are quite right! What you do not know yet is, that behind the noise hides a wonderful wave $f_{\text{true}}$, composed from two tones:
"""

# ╔═╡ 22d396c0-f6cf-11ea-04f4-0d54157e120e
f_true(x) = sin.(2 .* x) + sin.(3 .* x) + 1 ./ ( 1 .+ x )

# ╔═╡ 40e01698-f6cf-11ea-2a26-05d80ff61c90
begin
	t = collect(0:.1:2π)
	plot(t, f_true(t), label = "The true signal", lw = 2)
end

# ╔═╡ fbea7516-f6d2-11ea-22c0-8d0822eaf1b6
md"""
What we are going to do in this example, is to fit a curve to our noisy observations, which hopefully approximates the true source better, the more observations we can obtain.
Still, unfortunately, you do not know the true function (provided you did not peek at the hidden code cells of this notebook) - this is what we are going to change now. We will start by making  the wild assumption that if we were to select a finite set of points on the true curve, then these points are jointly Gaussian distributed with zero mean (for our convenience, not a general assumption) and a covariance matrix defined via a kernel function $k(x, x')$. The choice of this kernel function incorporates our belief into how smooth the true function is. Is the true function behind our radio signal a very smooth function? An RBF-kernel might be good for our problem. Is it not so smooth? Well...let's assume it is quite smooth. 
At this point you might be thinking: Hey, I thought this was all about coding some Julia, yet all I did for the past 5 minutes was reading about some noisy radio signal and a bit of statistics. Alright, alright, here you go: Let us start by realizing that our observation is a finite set of points, i.e. they should be jointly Gaussian with zero mean. How do we compute the covariance matrix? By arranging the kernel-values between each pair of points into a $N \times N$ matrix, where $N$ is the number of observations:
"""

# ╔═╡ c4d0f8c6-f6d6-11ea-309c-83ef468b7cac
md"""
There is quite a bit to unpack in these two lines, in the most literal sense of the word. If we look at the function $K(x, x')$, which gives us the kernel matrix, we assume that the inputs $x$ and $x'$ are two vectors of one-dimensional observations (which can be generalized but is what we want for our concrete example). Then first we turn them into a matrix of tuples containing all possible pairs of observations between $x$ and $x'$. The kernel function (specifically the RBF-kernel) takes a tuple as its input and unpacks this when calling the squared-euclidean-distance function. The result of $K(x, x)$ is the covariance matrix of our finite set of observations, as per our definition of GPs. Let us first have a look at it (*obs* is the signal we received and *x* are the corresponding points on the x-axis - both are defined in one of the hidden cells above):
"""

# ╔═╡ ca6d4cfa-f6d9-11ea-3426-e7c5836f3c04
md"""
## Posterior Predictive Distribution

Ok, that's nice, we now know everything about the jointly distributed observations that we need to do inference. The goal for now is to find a way to express our estimate of the true function in terms of its mean and variance. The mean will give us a set of points on the y-axis, corresponding to some points on the x-axis (i.e. a prediciton for $f_{\text{true}}(x)$) and the variance will give us the error at each point. The whole thing will result in a pretty neat plot in the end, but let's focus on the inference first.

So this is going to be a bit tricky: Let $f_*$ denote the function values we want to predict based on some new data $X_*$. Let further $y$ be the observations. We write the joint Gaussian between $y$ and $f_*$ in terms of their mean (i.e. zero) and their covariance matrix (which can be obtained in terms of the kernel function). This is standard probability theory - and I will not bore you with it if you are just here for the numerics! But if you care, and I dare say it is quite sweet to study, [Chapter 2.2. in this book (Equations 2.20 through 2.24)](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) constructs the joint and resolves it to the conditional, i.e. the posterior w.r.t $f_*$. And voilà: We have a formula for mean predicitons and the variance. These results are the posterior predicitve distribution of the observations and are obtained in a lot of standard literature. Here they are:
"""

# ╔═╡ 20071ac6-f6dd-11ea-1548-918bd22ee1af
md"""
$\overline{f_*} = K(X_*, X) ( K(X, X) + \sigma^2 I )^{-1} y,$
"""

# ╔═╡ 5f810c52-f6dd-11ea-00a0-d5d753bf7482
md"""
$\text{cov}(f_*) = K(X_*, X_*) - K(X_*, X) ( K(X, X) + \sigma^2 I )^{-1} K(X, X_*).$
"""

# ╔═╡ b0c39148-f6dd-11ea-3279-1de66377ed9d
md"""
As you can observe the above formulas contain a variance $\sigma^2$, which is the variance of *additive Gaussian noise* we assume on the data.

The rest of the example will go about as follows: First we will implement the above formulas given our aldready defined functions for kernels and afterwards we will play around with the hyperparameters to find the best fit to our curve-fitting problem. Variable $x$ denotes the vector of x-values of the observations, $y$ the corresponding y-values and $z$ a vector with new points on the x-axis that we would like to predict (i.e. find the corresponding y-values for).
"""

# ╔═╡ bffe8bf2-f734-11ea-2731-a70c5018a8a8
md"""
Let us start with one hundred random values on the x-axis:
"""

# ╔═╡ cc06952a-f734-11ea-2a6a-778b70adfbe0
z = sort(2π .* rand(100))

# ╔═╡ e51abc58-f734-11ea-07ae-a9331f3dbc31
md"""
Now let us just guess a fairly reasonable variance for the noise and plug everything in:
"""

# ╔═╡ 583411de-f736-11ea-1d31-f520c4b512b0
md"""
In the above plot you see the predicted curve for 100 sample points (feel free to play around with the number! What happens with only 5 samples? What changes when we move to 10000?) on the x-axis, together with the true curve and the actually predicted points. Around these points you'll see a ribbon, which indicates the standard deviation of each prediction, obtained from the *cov* function, defined above.
Plotting the graph together with the true function lets us observe that we are already roughly approximating the actual function, however, much more can be done if we properly adjust the hyperparameters of our model. 'Which hyperparamters?', you might ask. Well...for one we have the width of our RBF-kernel which is by default set to $2.0$. Secondly, if you remember, our choice of the noise-variance in the previous plot was also a simple guess. In the following we code two sliders that control the two parameters. Feel free to play around with them to find an optimal fit. Remember, however, that usually we do not know the original function, and thus, adequate methods for determining hyperparamters from training data must be applied for this. This, though, is beyond the scope of this example.
"""

# ╔═╡ a7c2092a-f738-11ea-23be-7d589006f20f
@bind noise_variance Slider(0:0.01:1)

# ╔═╡ 049e36e4-f73b-11ea-265a-6bc3a1027e00
@bind kernel_width Slider(0.001:0.001:3.0)

# ╔═╡ 17dfc200-f6d6-11ea-3543-ad8859406df1
begin
	rbf_kernel(y, σ²=2.0) = exp( -(1/σ²) .* sqeuclidean(y...) )
	K(x, x_) = rbf_kernel.(collect(Base.product(x, x_)), kernel_width)
end

# ╔═╡ 5b52bde6-f733-11ea-2860-df7f6349da21
begin
	μ(x, y, z, σ²) = K(z,x) * ( K(x,x) + σ² .* I)^-1 * y
	cov(x, z, σ²) = K(z,z) - K(z,x) * ( K(x,x) + σ² .* I)^-1 * K(x,z)
end

# ╔═╡ acabc8a4-f73d-11ea-18ec-0bb821fcd0c4
md"""
## Feeding More Data
And that is it! Here you have the toolset to denoise this old radio's signal. At last, you may want to see whether our predictions actually get better with more data. To do this, feel free to change the below slider. It controls the number of samples we can obtain from the radio signal. Pull the slider up slowly and observe how the curve changes with fixed hyperparamters.
"""

# ╔═╡ 31b76c3c-f73f-11ea-344c-ff9133f5f02b
@bind n_radio_samples Slider(64:1:10000)

# ╔═╡ f5d701a0-f6d0-11ea-1bee-f724743a62df
begin
	x = sort(2π .* rand(n_radio_samples))
	obs = f_true(x) .+ rand(Normal(0.0, .2), size(x, 1))
	plot(x, obs, label = "Transcribed signal", lw = 2)
end

# ╔═╡ 5d9fa2ee-f6d9-11ea-2988-db6afa9d3fba
K(x, x)

# ╔═╡ 0155cf3e-f735-11ea-2eab-0fae8eacc7e4
begin
	plot(z, μ(x, obs, z, 0.2), label="The predicted curve", lw=2)
	plot!(t, f_true(t), label="The true curve", lw=2)
	scatter!(
		z, μ(x, obs, z, 0.2), 
		ribbon=sqrt.(diag(cov(x, z, 0.2))), 
		label="Predicted points"
	)
end

# ╔═╡ 06ec2ecc-f738-11ea-2036-cf16771f69a1
begin
	plot(z, μ(x, obs, z, noise_variance), label="The predicted curve", lw=2)
	plot!(
		t, 
		f_true(t), 
		title="Noise-variance: $noise_variance, Kernel-width: $kernel_width", 		
		label="The true curve", lw=2
	)
end

# ╔═╡ 9a66a6c2-f761-11ea-0691-d3a5120db494
md"""
## Further Reading
If this notebook has got you all excited about Gaussian Processes, you should definitely check out the [Rasmussen and Williams Book]() that I referenced in the beginning. It contains all the maths you need and discusses GPs from different perspectives in the light of machine learning. Maybe, when changing the slider above, you realized that the computation takes quite a while for a high value of *n\_radio\_samples*! This is mostly due to the matrix inversion that we perform. There is, however, a neat trick to mitigate this using a cholesky decompostion. It is discussed in the book as well. If you want to test you skills, adjust the respective cell in this notebook with a more performant implementation of the GP-posterior.

If you want to have more hands-on coding examples, check out [this blog](https://katbailey.github.io/post/gaussian-processes-for-dummies/) by Katherine Bailey, that I found very interesting.

Another great blog to study is [this one](http://krasserm.github.io/2018/03/19/gaussian-processes/) by Martin Krasser - it goes far more in-depth than this notebook and is an awesome resource on the topic.
"""

# ╔═╡ Cell order:
# ╟─e125117a-f75c-11ea-316c-87d98ff4fd33
# ╠═1186151e-f6cf-11ea-1c3c-13083a1768e8
# ╠═1626d748-f6cf-11ea-16cf-ad8d3458005c
# ╟─de5dddac-f75d-11ea-0943-27cd853abb42
# ╟─a302f556-f6d0-11ea-09b7-1b41dd4c9d1e
# ╟─f5d701a0-f6d0-11ea-1bee-f724743a62df
# ╟─ee1dc330-f6d1-11ea-1164-cbef833a75a2
# ╟─22d396c0-f6cf-11ea-04f4-0d54157e120e
# ╟─40e01698-f6cf-11ea-2a26-05d80ff61c90
# ╟─fbea7516-f6d2-11ea-22c0-8d0822eaf1b6
# ╠═17dfc200-f6d6-11ea-3543-ad8859406df1
# ╟─c4d0f8c6-f6d6-11ea-309c-83ef468b7cac
# ╠═5d9fa2ee-f6d9-11ea-2988-db6afa9d3fba
# ╟─ca6d4cfa-f6d9-11ea-3426-e7c5836f3c04
# ╟─20071ac6-f6dd-11ea-1548-918bd22ee1af
# ╟─5f810c52-f6dd-11ea-00a0-d5d753bf7482
# ╟─b0c39148-f6dd-11ea-3279-1de66377ed9d
# ╠═5b52bde6-f733-11ea-2860-df7f6349da21
# ╟─bffe8bf2-f734-11ea-2731-a70c5018a8a8
# ╠═cc06952a-f734-11ea-2a6a-778b70adfbe0
# ╟─e51abc58-f734-11ea-07ae-a9331f3dbc31
# ╠═0155cf3e-f735-11ea-2eab-0fae8eacc7e4
# ╟─583411de-f736-11ea-1d31-f520c4b512b0
# ╠═a7c2092a-f738-11ea-23be-7d589006f20f
# ╠═049e36e4-f73b-11ea-265a-6bc3a1027e00
# ╠═06ec2ecc-f738-11ea-2036-cf16771f69a1
# ╟─acabc8a4-f73d-11ea-18ec-0bb821fcd0c4
# ╠═31b76c3c-f73f-11ea-344c-ff9133f5f02b
# ╟─9a66a6c2-f761-11ea-0691-d3a5120db494
