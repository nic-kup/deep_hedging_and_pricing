# Deep Hedging and pricing for non continuous payoffs with transaction costs.

This code implements the methodology presented in:

* https://arxiv.org/abs/1802.03042
* https://arxiv.org/abs/1804.05394
* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3355706
* https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf


Still some missing.

### Requirements

The code is developed using Python 3.7.6. To install further Python dependencies:

    $ pip install -r requirements.txt 
    
## Structure
```
Recreation
│   README.md
|   bs_vanilla_call_option.py
|   bs_digi_option_tc_static.py
|   optimal_stopping.py
|   bs_vanilla_american_option.py
│    ... Other Example files
│
└─── backend
│   │   BaseModel.py        # a base model for market models
│   │   custom_layers.py    # custom keras layers
│   │   neural_network.py   # all the neural network structures used for hedging / pricing
|   |   pricer.py           # functions which calculate closed form solutions for pricing
|   |   simulation.py       # market models (inherit from BaseModel)
|   |   neural_network_recurrent.py # Trying to build a LSTM version
|
└─── test
|   |   ... Contains testing files
```
## Example

### Vanilla Call option pricing
To better understand this library first look at the example file `bs_vanilla_call_option.py` which
prices a call option in a market without transaction costs.

First we import `backend.neural_network` and `backend.pricer`, the libraries for the neural network models
and classical pricing methods respectively. This will also automatically import `backend.simulation` which
contains the models for generating stock prices. Currently it has a model for the classical Black Scholes
model and the Heston model.

After defining some parameters we define the Black Scholes model by calling the class `BlackScholesModel`. This model
allows us to generate n-dimensional Geometric Brownian Motions. It inherits its structure from `BaseModel`, as does
`HestonModel`, which also lives in `backend.simulation`.

Next we define our learner `SimpleLearner`. It is a simple model which consists of one neural network per time step
which only takes the stock price. 

We are now ready to define a basic training loop. To do this we must first create a list of learning rates. We then
 compile the model using the optimizer Adam. We then generate the needed data using our Object `BM`. The X will be
  the simulated stock price, and y will just be zeros. Then, since `model` inherits from `tf.keras.Model` we can just
   use the inbuilt `fit` function.
   
   After our training is complete we just do some plotting to better understand our model.
   
### Digital option with transaction costs
Now that we understand this basic example let us look at the file `bs_digi_call_option_tc_static.py`. Again we import
`backend.neural_network` and `backend.pricer` and define out Black Scholes Model `BM`. Next we define functions
relating to our digital option, cost function, utility function and loss.
  
In this example the price **can not** be learned dynamically as in the previous example (for now, maybe). So we
initialize a list of possible prices.
   
Next we initialize our hedging model `Learner` which is very similar to `SimpleLearner`, however it takes an
additional input at every timestep, namely the previous output. For the following training loop we specify not only
learning rates, but also for how many epochs the model should be trained with that learning rate. In this particular
training loop we start with small datasets and then double them each iteration. This was implemented to help with
exploding losses early in training. This was also helped by making the initialized weights smaller.
   
Once our model has trained we re-train for each new price. This is reasonable under the assumption that a close to
optimal strategy should be able to be found reasonably close the previous strategy with the previous fixed price.
After having trained for every price we choose the price closest to zero.