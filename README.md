# HackTheBurgh

# An Analysis of and FPGA Implementation for Spectral Analysis of Bats

We implemented an FPGA system to classify in realtime resevoir bat species by their hidden echolocation frequencies using a decision tree

## Project Description
### Epidemiological Applications
There are a number of bat species in specific regions in the world which harbour fatal disease-causing viruses such as Ebola, Marburg, Nipah and various Coronaviruses. These bat species are known as resevoirs. It's important to understand the distribution of these species in order to predict the risk of spillover events into human populations. Consequently developing a low cost and efficient solution to identifying bat species based on ultrasonic call data is a promising strategy for early detection programmes. 

There are certain bat species in certain locations that carry risk. As such, if we can classify bats based on their ultrasonic calls inaudible to humans, in a low-cost, high volume and low power fashion, we can distribute this device to regions at high risk of disease as an effective warning system.

### Ecological Applications
As bats make up 20% of all mammalian species, understanding the effect of anthropogenic climate change and human land use on their distribution is paramount to the success of wildlife conservation efforts. Additionally, identification of species with a high risk of displacement will enable more responsible policy and urban planning decisions.

It's important to do background research on resevoir species and determine which ones in which regions are most likely to carry these diseases. We have identified *Pteroptera spp.*, *Rhinolophus spp.*, and *Rousettus spp.* as reservoirs of Henipaviruses, Coronaviruses, and Marburg viruses, respectively. These species can be found around the world but are concentrated in key areas of concern such as Western Africa and South/Southeast Asia. By training our model on accessible data from European relatives, we will present a proof-of-concept system that could be applied to these regions for public health surveillance and ecological applications.

### Implementation Details
We built a real-time bat call classifier that runs entirely in FPGA logic. It listens to ultrasound at 256 kHz, picks out each echolocation call, and identifies the species, flagging bats known to carry risk as identified above. It was trained on a subset of european bats as a proof of concept for bat classification based on unique sound profiles. It achieves an accuracy of 82% across 12 classes. 

The classifier uses about 4% of a Zynq-7020’s logic and runs 11× faster than real time, with an estimated draw of around 11 mW as per Vivado simulation. The design is portable enough to move onto a low-cost FPGA or a custom chip. Networks of these sensors could sit at cave entrances and along migration routes, showing where reservoir bats roost and how they move with the seasons. Retargeting to bats in South East Asia, Latin America or Africa only needs labelled recordings, not new hardware. It identifies species, not infection, and it gives health and wildlife agencies information they currently don’t have.


# ChirosetEurope: A Curated Bioacoustics Dataset of European Bat Vocalizations
- [Preprint Link](https://ecoevorxiv.org/repository/view/14543/)
- [Dataset Link](https://zenodo.org/records/20773227)
- Citation: (Baier *et al*., 2026)

