# Predictive coding model

I came across this youtube video, which discusses an interesting architecture for a neural network. I'd like to take a crack an implementing a CPU version of the idea, in rust. If this works at all (i.e. if I can get it to recognise and generate handwritten numbers) then I'll see if I can do a GPU implementation as well.


# Plotting

I am running this in a devcontainer, on my server, connected to via SSH, from a windows machine. The remote stuff is all set up, but two things need to be sorted out on the client side:

1. The SSH needs to have window forwarding enabled (e.g. `ssh -Y`). To do this properly in the ssh config file,
```
		ForwardAgent yes
		ForwardX11 yes
		ForwardX11Trusted yes
```
2. Install [VcXsrv](https://github.com/marchaesen/vcxsrv/releases) on the windows machine, to be the window server. Make sure it's running!

To test that the graphs will work, run `make test-liveplot`. You should get a sine wave plotted, and it should update over time. The framerate might be horrid, though.

TODO: Set up a server and client, so we can stream data instead of the whole window.
