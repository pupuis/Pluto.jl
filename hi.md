Hey!

I was curious of our forks and then I found this! So cool that you are rewriting to TypeScript - I would love to work in TS! I did considere TS before, but I had some concerns. Maybe you could comment on these!

I thought that it is more important for Pluto to be buildless, because installing Pluto has to be super fast and easy - so no installing Node and building on the user's machine. This means that we need to build ourselves and ship the "binairies" to the user. I think that the only way to do that (without a complicated and fragile Julia package install process) is to just put all binairies inside the repo itself. That does not sound nice, but it's also no the end of the world. Do you see another way?

The other argument is subjective - I think that JavaScript is very powerful today and I hope that the future of web dev is buildless! It just feels nice and simple for the source files (.js) to be the thing running in the browser - this also makes it a lot simpler for people to make small tweaks and contribute to the project. (On the other hand, a TSC is built into VS Code, so maybe people can work on the project without even knowing it!) Also note that there is no need for minification or bundling - Pluto runs on localhost, and uses a service worker cache for remote dependencies.

Let me know what you think! It would be awesome if you can help me solve these two concerns. I would also be happy to video/audio call sometime: fonsvdplas@gmail.com

-Fonsi

(Don't merge this PR)
