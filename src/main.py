import analysis
import data_collection
import data_preprocessing 
import model_selection
import visualization

def process_tickers(tickers):
    
    raw_data = data_collection.download(tickers)
    
    clean_data = data_preprocessing.clean(raw_data)
    prepared_data = data_preprocessing.prepare(clean_data)
    input_shape = prepared_data.shape[-2:]
    
    lstm_model = model_selection.create_lstm(input_shape)
    
    lstm_model = model_selection.create_lstm(input_shape) 
    model = model_selection.train(lstm_model, prepared_data)
    
    trades = analysis.generate_trades(model, prepared_data)
    metrics = analysis.evaluate(trades)
    
    visualization.plot(prepared_data, model)
    visualization.plot_performance(metrics)
    
if __name__ == "__main__":
   
   tickers = ["AAPL", "TSLA"]
   process_tickers(tickers)