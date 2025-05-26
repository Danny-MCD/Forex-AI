#include <Trade\Trade.mqh>
CTrade trade;

void OnTick()
{
   double BuySignalBuffer[], SellSignalBuffer[];
   ArraySetAsSeries(BuySignalBuffer, true);
   ArraySetAsSeries(SellSignalBuffer, true);
   CopyBuffer(iCustom(_Symbol, PERIOD_M1, "YourIndicator"), 0, 0, 1, BuySignalBuffer);
   CopyBuffer(iCustom(_Symbol, PERIOD_M1, "YourIndicator"), 1, 0, 1, SellSignalBuffer);

   if(BuySignalBuffer[0] != EMPTY_VALUE)
      trade.Buy(0.1, _Symbol, Ask, Ask - 111 * Point(), Ask + 30 * Point());
   if(SellSignalBuffer[0] != EMPTY_VALUE)
      trade.Sell(0.1, _Symbol, Bid, Bid + 111 * Point(), Bid - 30 * Point());
}