#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

input string Python_Script = "predictor.py";  // Python script name

double BuySignalBuffer[];
double SellSignalBuffer[];

int OnInit()
{
   SetIndexBuffer(0, BuySignalBuffer, INDICATOR_DATA);
   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_ARROW);
   PlotIndexSetInteger(0, PLOT_ARROW, 233); // Up arrow
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, clrGreen);
   
   SetIndexBuffer(1, SellSignalBuffer, INDICATOR_DATA);
   PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_ARROW);
   PlotIndexSetInteger(1, PLOT_ARROW, 234); // Down arrow
   PlotIndexSetInteger(1, PLOT_LINE_COLOR, clrRed);
   
   return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(prev_calculated == rates_total) return(rates_total);

   double ema_fast = iMA(_Symbol, PERIOD_M1, 8, 0, MODE_EMA, PRICE_CLOSE, 1);
   double ema_slow = iMA(_Symbol, PERIOD_M1, 20, 0, MODE_EMA, PRICE_CLOSE, 1);
   double rsi = 50;  // Placeholder
   double macd = 0;  // Placeholder
   double bb_width = 0;  // Placeholder

   string data_point = DoubleToString(ema_fast) + "," + DoubleToString(ema_slow) + "," + 
                       DoubleToString(rsi) + "," + DoubleToString(macd) + "," + 
                       DoubleToString(bb_width);

   // Simulated Python call (replace with actual MQL5 Python API)
   string prediction_str = PythonRun(Python_Script, "predict_swing", data_point);  // Custom function
   string parts[];
   StringSplit(prediction_str, ',', parts);
   int prediction = StringToInteger(parts[0]);
   double probability = StringToDouble(parts[1]);

   if(prediction == 1)  // Swing high (sell)
   {
      SellSignalBuffer[rates_total-1] = high[rates_total-1] + 10 * Point();
      Alert("Sell Signal - Probability: " + DoubleToString(probability, 2));
      PlaySound("alert.wav");
   }
   else
      SellSignalBuffer[rates_total-1] = EMPTY_VALUE;

   if(prediction == 0)  // Swing low (buy)
   {
      BuySignalBuffer[rates_total-1] = low[rates_total-1] - 10 * Point();
      Alert("Buy Signal - Probability: " + DoubleToString(probability, 2));
      PlaySound("alert.wav");
   }
   else
      BuySignalBuffer[rates_total-1] = EMPTY_VALUE;

   return(rates_total);
}