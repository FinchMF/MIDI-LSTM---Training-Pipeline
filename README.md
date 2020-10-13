## TRAINING PIPELINE 

* The aim is to artifically write melodies like BACH. This pipeline will gather, transform and train.

### Contents of Pipeline:

* Web MIDI Scraper
* MIDI parser
* MIDI data transformer
* RNN-ATTN model with Stacked LSTMs 

#### Webscraper:

* MIDI is scraped from www.jsbach.net. All cello peices with single voice parts are scraped. 

#### MIDI Parser:

* All MIDI handling is done with Music21 tools

#### RNN-ATTN with Stacked LSTMs:

* Model's architeture below

        Embedded Pitch  +  Embedded Duration
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


## EXECUTE PIPELINE

### Aim of Pipeline:

The expection for the pipeline is to:
* webscrape midi 
* transform and store data as necessary
* construct NN architecture 
* engineer the data to achitecture input
* train model on scraped midi
* save weights of best trained model
* generate midi file

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



## Description 
* This script will setup the directory and initate the webscraper to populate the proper directories with the scrapped MIDI, as well as other data. Once the data is prepared, the script triggers the transformation and data engineering of the MIDI into LSTM inputs. The training process is then set in motion. The script concludes using the saved weight matrices to generate melodies from the matrices trained on BACH's data. The resulting MIDI file is then placed in the subdirectory, 'output', found in the 'Run Folder'

