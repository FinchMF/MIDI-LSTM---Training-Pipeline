## TRAINING PIPELINE 

### Contents of Pipeline:

* Web MIDI Scraper
* MIDI parser
* MIDI data transformer
* RNN-ATTN model with Stacked LSTMs 

#### Webscraper:

* MIDI is scraped from www.jsbach.net. All cello peices with single voice parts are scraped. We do this because the aim is to artifically write melodies like BACH. 

#### MIDI Parser:

* All MIDI handling is done with Music21 tools

#### RNN-ATTN with Stacked LSTMs:

* Model's architeture below

        Embedded Pitch + Embedded Duration
                        |
                        |
                        |
                        LSTM
                        |
                        LSTM
                        -Dropout
                        |
                        ATTN MODEL
                        |
                        Fully Connected Layer
                        |
                        |
        Fully Connected Layer (softmax output)
                |                       |
                |                       |
                Pitch Output            Duration Output


Parameters and Model Configurations, see: 

    composer_RNN.py 
    composer_params.py

## 


## EXECUTE PIPELINE

### Aim of Pipeline:

The expection for the pipeline is to:
* webscrape midi 
* transform and store data as necessary
* construct NN architecture 
* engineer the data to achitecture input
* train model on scraped midi
* save weights of best trained model

### How to Use
    
    git clone https://github.com/FinchMF/MIDI-LSTM---Training-Pipeline.git
    pip install -r requirements.txt

There are two operational shell scripts:

    $ bash reset.sh
    $ bash execute.sh

### RESET
Reset will ease folders: 'data' and 'run'. This resets repo.

### EXECUTE
After the initial repo clone, run: 
        
        $ bash execute.sh

This script will setup the directory and initate the webscraper to populate the porper directories with MIDI from a webpage. Once the data is prepared, the script tiggers the transformation and data engineering of the data into inputs. The training process is then set in motion. The script concludes with saved weight matrices that can then be used with future instantiated models to generate melodies from the matrices trained on BACH's data. 