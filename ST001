#include <MetaTrader5.h>

// 定义用于存储历史价格数据（收盘价）的数组，用于训练深度学习模型
double historicalPrices[];
// 定义用于存储训练标签（比如未来一段时间后的价格变化方向或者幅度等，示例简单化处理）的数组
double trainingLabels[];

// 定义LSTM网络的相关参数
int inputSize = 15;  // 增大输入特征维度，可结合更多技术指标等，根据实际调整
int hiddenSize = 30; // 调整LSTM隐藏层神经元数量，优化网络学习能力
int outputSize = 1;  // 输出维度（比如预测价格是上涨还是下跌，简单示例）
int numLayers = 2;    // LSTM网络层数

// 结构体定义LSTM单元，用于存储单元内的各种状态和参数
struct LSTMCell
{
    double* inputGate;
    double* forgetGate;
    double* outputGate;
    double* cellState;
    double* hiddenState;
    double** weightsInput;  // 输入到LSTM单元的权重
    double** weightsHidden; // 隐藏层到LSTM单元的权重（如果是多层LSTM）
    double* biases;
};

// 初始化LSTM单元的权重和偏置等参数（简单随机初始化示例，可优化）
void InitializeLSTMCell(LSTMCell& cell, int inputSize, int hiddenSize)
{
    cell.inputGate = new double[hiddenSize];
    cell.forgetGate = new double[hiddenSize];
    cell.outputGate = new double[hiddenSize];
    cell.cellState = new double[hiddenSize];
    cell.hiddenState = new double[hiddenSize];
    cell.weightsInput = new double*[hiddenSize];
    cell.weightsHidden = new double*[hiddenSize];
    cell.biases = new double[hiddenSize];

    for (int i = 0; i < hiddenSize; i++)
    {
        cell.weightsInput[i] = new double[inputSize];
        cell.weightsHidden[i] = new double[hiddenSize];
        for (int j = 0; j < inputSize; j++)
        {
            cell.weightsInput[i][j] = MathRand() * 0.1 - 0.05;  // 随机初始化权重在一定范围内
        }
        for (int k = 0; k < hiddenSize; k++)
        {
            cell.weightsHidden[i][k] = MathRand() * 0.1 - 0.05;
        }
        cell.biases[i] = MathRand() * 0.1 - 0.05;
    }
}

// LSTM单元的前向传播计算逻辑（简化示意，基于基本的LSTM数学原理）
void LSTMCellForwardPropagation(LSTMCell& cell, double* input, double* prevHiddenState)
{
    int hiddenSize = ArraySize(cell.hiddenState);
    int inputSize = ArraySize(input);

    // 计算输入门
    for (int i = 0; i < hiddenSize; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputSize; j++)
        {
            sum += cell.weightsInput[i][j] * input[j];
        }
        for (int k = 0; k < hiddenSize; k++)
        {
            sum += cell.weightsHidden[i][k] * prevHiddenState[k];
        }
        sum += cell.biases[i];
        cell.inputGate[i] = 1 / (1 + MathExp(-sum));
    }

    // 计算遗忘门（类似输入门计算逻辑）
    for (int i = 0; i < hiddenSize; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputSize; j++)
        {
            sum += cell.weightsInput[i][j] * input[j];
        }
        for (int k = 0; k < hiddenSize; k++)
        {
            sum += cell.weightsHidden[i][k] * prevHiddenState[k];
        }
        sum += cell.biases[i];
        cell.forgetGate[i] = 1 / (1 + MathExp(-sum));
    }

    // 计算候选细胞状态
    double* candidateCellState = new double[hiddenSize];
    for (int i = 0; i < hiddenSize; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputSize; j++)
        {
            sum += cell.weightsInput[i][j] * input[j];
        }
        for (int k = 0; k < hiddenSize; k++)
        {
            sum += cell.weightsHidden[i][k] * prevHiddenState[k];
        }
        sum += cell.biases[i];
        candidateCellState[i] = MathTanh(sum);
    }

    // 更新细胞状态
    for (int i = 0; i < hiddenSize; i++)
    {
        cell.cellState[i] = cell.forgetGate[i] * cell.cellState[i] + cell.inputGate[i] * candidateCellState[i];
    }

    // 计算输出门
    for (int i = 0; i < hiddenSize; i++)
    {
        double sum = 0;
        for (int j = 0; j < inputSize; j++)
        {
            sum += cell.weightsInput[i][j] * input[j];
        }
        for (int k = 0; k < hiddenSize; k++)
        {
            sum += cell.weightsHidden[i][k] * prevHiddenState[k];
        }
        sum += cell.biases[i];
        cell.outputGate[i] = 1 / (1 + MathExp(-sum));
    }

    // 更新隐藏状态
    for (int i = 0; i < hiddenSize; i++)
    {
        cell.hiddenState[i] = cell.outputGate[i] * MathTanh(cell.cellState[i]);
    }

    delete[] candidateCellState;
}

// 整个LSTM网络的前向传播，通过依次调用LSTM单元的前向传播实现（假设多层情况）
double* LSTMNetworkForwardPropagation(LSTMCell* lstmCells, double** inputData, int sequenceLength)
{
    int numLayers = ArraySize(lstmCells);
    int hiddenSize = ArraySize(lstmCells[0].hiddenState);

    double** hiddenStates = new double*[sequenceLength];
    for (int t = 0; t < sequenceLength; t++)
    {
        hiddenStates[t] = new double[hiddenSize];
    }

    for (int layer = 0; layer < numLayers; layer++)
    {
        if (layer == 0)
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                LSTMCellForwardPropagation(lstmCells[layer], inputData[t], new double[hiddenSize]);  // 初始隐藏状态可设为0向量等情况
                hiddenStates[t] = lstmCells[layer].hiddenState;
            }
        }
        else
        {
            for (int t = 0; t < sequenceLength; t++)
            {
                LSTMCellForwardPropagation(lstmCells[layer], hiddenStates[t], hiddenStates[t - 1]);
                hiddenStates[t] = lstmCells[layer].hiddenState;
            }
        }
    }

    // 最后可以根据需要从隐藏状态获取最终的网络输出（示例简单返回最后一个时间步的隐藏状态，可根据实际调整）
    return hiddenStates[sequenceLength - 1];
}

// 定义出场模式枚举
enum ExitMode { IMMEDIATE_EXIT = 1, NEXT_BAR_EXIT = 0 };

// 外部参数定义，添加更清晰的注释说明用途
string Introduce = "=========介绍=========";   // 用于展示相关介绍信息
string Version = "3.30";   // 版本号
string CompatibleSymbol = "XAUUSD,EURUSD,GBPUSD,GBPJPY,USDJPY,NZDJPY,AUDNZD...";   // 适用交易品种列表
string CompatibleTimeframe = "M15";   // 适用的交易周期
string TutorialLinks = "https://www.eabook.cn/ea-224-1-1.html";   // 参数与教程链接
string SignalLink = "https://www.mql5.com/zh/signals/396812";   // 信号链接
string Basic = "=========基本设置=========";   // 基本设置相关信息展示
double BasicEquity = 300;    // 参考净值，用于计算开仓手数等
double Lots = 0;    // 净值适应手数，若小于等于0则使用初始开仓手数
double InitLots = 0.01;    // 开仓手数初始值
double Spread = 3;    // 允许的最大点差
int PointX = 10;    // 大点与小点比值，用于价格等相关计算调整
string TrendJud = "=========趋势箱体=========";   // 趋势箱体相关信息展示
int TrendPeriod = 66;    // 趋势箱体周期
double TrendGap = 60;    // 当前价与顶/底的距离，用于判断回调情况
string OscJud = "=========震荡箱体=========";   // 震荡箱体相关信息展示
int OSCPeriod = 9;    // 震荡箱体周期
double OSCGap = 12;    // 允许的最小箱体尺寸相关参数
string EnterJud = "=========进场=========";   // 进场相关信息展示
double OSCLevel = 10;    // 开仓水平位参数
string ExitJud = "=========出场=========";   // 出场相关信息展示
ExitMode ExitSelect = IMMEDIATE_EXIT;    // 出场时机选择，立即出场或下根K线出场
double DirectCloseLevel = 12;    // 平仓水平位参数
double ProfitPoint = 20;   // 盈利点数，用于判断是否基于盈利点数平仓
int SLPeriod = 66;  // 止损周期，用于计算止损相关价格
double MinSL = 25;    // 最小止损点数
double MaxSL = 1500;  // 最大止损点数
string Time_Filter = "=========交易时间=========";   // 交易时间相关信息展示
string TimeDes = "这里设定的是平台时间，可以根据需要自由调整。";   // 时间设置说明
int startHourLimit = 24;  // 重命名原'总_1_in'，表示时间限制相关的小时数
int startMinuteLimit = 1;  // 重命名原'总_2_in'，表示时间限制相关的分钟数
int StartHour = 0;    // 开始开仓（小时）
int StartMinute = 0;    // 开始开仓（分钟）
int StopHour = 24;    // 停止开仓（小时）
int StopMinute = 0;    // 停止开仓（分钟）
int FridayProhibitStartHour = 24;    // 周五停止开仓（小时）
int FridayProhibitStartMinute = 0;    // 周五停止开仓（分钟）
string ColorFlage = "=========颜色标记=========";   // 颜色标记相关信息展示，用于区分多空单等
color LongColor = clrBlue;    // 多单颜色
color ShortColor = clrRed;    // 空单颜色
string OtherFlage = "=========其他=========";   // 其他相关信息展示
int MAGIC = 8888888;    // 订单识别码/魔术码（需与其他EA设置不同），用于区分不同EA的订单
string Author = "eabook]SuperTrend";   // 订单注释，用于标识订单来源等信息

// 以下变量用于控制出场、开仓时机等，添加更清晰注释说明用途
int chartBarCount = 0;  // 此变量用来控制出场时机，记录K线数量
double chartBarCountForBuy = 0.0; // 用于控制同一根K线只开一次多单
double chartBarCountForSell = 0.0; // 用于控制同一根K线只开一次空单

// 定义训练数据的窗口大小（即每次使用多少历史数据点进行训练，可根据实际调整）
int trainingWindowSize = 300;
// 定义重新训练模型的间隔（多少根K线后重新训练，可调整）
int retrainInterval = 150;
// 计数器，用于记录K线数量，判断何时重新训练模型
int trainingCounter = 0;

// 新增变量，用于记录模型预测的置信度，辅助交易决策
double predictionConfidence = 0.0;

// 初始化函数，进行相关初始化操作，包括MT5平台初始化、数据数组初始化、LSTM网络初始化等
int OnInit()
{
    // 初始化MT5平台连接
    if (!MetaTrader5::Initialize())
    {
        Print("初始化MT5失败");
        return -1;
    }

    // 初始化历史价格和训练标签数组（示例简单设置为固定大小，实际可动态分配等）
    ArrayResize(historicalPrices, trainingWindowSize);
    ArrayResize(trainingLabels, trainingWindowSize);

    // 初始化LSTM网络的各个LSTM单元
    LSTMCell* lstmCells = new LSTMCell[numLayers];
    for (int i = 0; i < numLayers; i++)
    {
        InitializeLSTMCell(lstmCells[i], inputSize, hiddenSize);
    }

    // 数据标准化相关参数初始化（示例简单用均值和标准差，后续可优化）
    double meanPrice = 0;
    double stdDevPrice = 0;
    InitializeDataStatistics(historicalPrices, meanPrice, stdDevPrice);

    // 这里可以添加代码从历史数据中获取初始的价格数据填充数组，按照MT5的数据获取方式来实现，并进行标准化处理
    FillInitialHistoricalData(historicalPrices, meanPrice, stdDevPrice);

    return 0;
}

// 每一次行情变动（新的报价到来）时触发的函数
void OnTick()
{
    bool isFridayNoTrade = false;
    // 判断服务器是否允许EA交易、客户端是否开启EA交易、智能交易是否中止、交易服务器是否繁忙，若有任一条件不满足则直接返回，不进行后续操作
    if (!MetaTrader5::TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ||!MetaTrader5::ExpertEnable() || MetaTrader5::IsStopped() || MetaTrader5::TerminalInfoInteger(TERMINAL_SERVER_BUSY))
        return;

    // 更新历史价格数据，移除最早的数据，添加最新的收盘价，按照MT5获取数据方式调整，并进行标准化处理
    UpdateHistoricalData(historicalPrices, meanPrice, stdDevPrice);

    // 增加训练计数器
    trainingCounter++;

    // 判断是否达到重新训练模型的间隔
    if (trainingCounter >= retrainInterval)
    {
        // 更精细地准备训练数据和标签，结合更多交易相关信息生成标签（示例可改进），并进行相应的数据标准化处理
        PrepareTrainingDataAndLabels(historicalPrices, trainingLabels, meanPrice, stdDevPrice);

        // 使用训练数据和标签训练LSTM网络，并获取训练后的模型预测置信度，完善训练逻辑，包括优化器选择等（示例简单用随机梯度下降，可改进）
        predictionConfidence = TrainLSTMNetwork(historicalPrices, trainingLabels, trainingWindowSize, meanPrice, stdDevPrice);

        trainingCounter = 0;
    }

    // 获取当前输入数据，丰富输入特征，不仅仅局限于收盘价（示例可扩展），同时对输入数据进行标准化
    double** inputData = PrepareCurrentInputData(historicalPrices, meanPrice, stdDevPrice);

    // 通过LSTM网络进行预测（得到预测结果，示例简单假设输出为价格上涨或下跌的概率等）
    double* prediction = LSTMNetworkForwardPropagation(lstmCells, inputData, 1);

    // 根据模型预测结果以及其他交易规则判断开仓、平仓等操作

    // 根据出场时机选择进行平仓操作
    if (ExitSelect == NEXT_BAR_EXIT) // 出场时机为下根K线出场
    {
        if (chartBarCount!= Bars) // 表示新增了一根K线，即下根K线
        {
            CloseBuyOrders();
            CloseSellOrders();
            chartBarCount = Bars; // 记录K线数
        }
    }
    else                    // 出场时机为立即出场
    {
        CloseBuyOrders();
        CloseSellOrders();
        chartBarCount = Bars;
    }

    // 判断是否为周五，若为周五则进一步判断是否超出周五不开仓时间范围
    if (TimeDayOfWeek(TimeCurrent()) == 5)
    {
        isFridayNoTrade = false;
        if (IsFridayNoTradeTime(TimeCurrent()))
        {
            isFridayNoTrade = true;
        }
        if (isFridayNoTrade)
            return;   // 超出周五设定开仓时段则不再开仓，避免持仓过周末
    }

    	// 根据K线数量控制开多单逻辑，同一根K线只开一次多单，结合模型预测结果及置信度等多因素判断
	if (chartBarCountForBuy!= Bars && prediction[0] > 0.5 && predictionConfidence > 0.6 && IsValidBuySignal())
	{
		OpenBuyOrder();
		chartBarCountForBuy = Bars;
	}

	// 根据K线数量控制开空单逻辑，同一根K线只开一次空单，结合模型预测结果及置信度等多因素判断
	if (chartBarCountForSell!= Bars && prediction[0] < 0.5 && predictionConfidence > 0.6 && IsValidSellSignal())
	{
		OpenSellOrder();
		chartBarCountForSell = Bars;
	}
}

// 智能交易关闭时触发的函数，此处可进行一些清理资源等操作，比如释放LSTM网络相关内存
void OnDeinit(const int reason)
{
    // 释放LSTM网络中动态分配的内存
    for (int i = 0; i < numLayers; i++)
    {
        LSTMCell& cell = lstmCells[i];
        delete[] cell.inputGate;
        delete[] cell.forgetGate;
        delete[] cell.outputGate;
        delete[] cell.cellState;
        delete[] cell.hiddenState;
        for (int j = 0; j < hiddenSize; j++)
        {
            delete[] cell.weightsInput[j];
            delete[] cell.weightsHidden[j];
        }
        delete[] cell.weightsInput;
        delete[] cell.weightsHidden;
        delete[] cell.biases;
    }
    delete[] lstmCells;
}

// 判断当前时间是否处于周五不开仓的时间区间内
bool IsFridayNoTradeTime(int currentTimeInSeconds)
{
    int fridayProhibitStartTimeInSeconds = FridayProhibitStartHour * 3600 + FridayProhibitStartMinute * 60;
    int startLimitTimeInSeconds = startHourLimit * 3600 + startMinuteLimit * 60;
    return ((currentTimeInSeconds >= fridayProhibitStartTimeInSeconds && currentTimeInSeconds <= startLimitTimeInSeconds) ||
            (fridayProhibitStartTimeInSeconds > startLimitTimeInSeconds && ((currentTimeInSeconds >= fridayProhibitStartTimeInSeconds && currentTimeInSeconds <= 24 * 3600 - 1) ||
                                                                       (currentTimeInSeconds >= 0 && currentTimeInSeconds <= startLimitTimeInSeconds))));
}

// 判断当前时间是否处于允许交易的时间区间内
bool IsInTradingTime(int currentTimeInSeconds, int startHour, int startMinute, int stopHour, int stopMinute)
{
    int startTimeInSeconds = startHour * 3600 + startMinute * 60;
    int stopTimeInSeconds = stopHour * 3600 + stopMinute * 60;
    return (currentTimeInSeconds >= startTimeInSeconds && currentTimeInSeconds <= stopTimeInSeconds);
}

// 计算止损价，考虑最小、最大止损限制，同时结合当前市场波动情况（示例简单通过ATR指标体现，可优化）
double CalculateStopLossPrice(double lowPrice, double bidPrice, double minSL, double maxSL, int pointX, double point, double atrValue)
{
    double stopLossPrice = NormalizeDouble(lowPrice, Digits);
    // 根据ATR动态调整止损距离，这里简单示例按一定比例系数调整，可根据实际优化
    double adjustedMinSL = minSL * (1 + atrValue / 100);
    double adjustedMaxSL = maxSL * (1 + atrValue / 100);
    if (stopLossPrice > bidPrice - adjustedMinSL * pointX * point)
    {
        stopLossPrice = bidPrice - adjustedMinSL * pointX * point;
    }
    else
    {
        if (stopLossPrice < bidPrice - adjustedMaxSL * pointX * point)
        {
            stopLossPrice = bidPrice - adjustedMaxSL * pointX * point;
        }
    }
    return NormalizeDouble(stopLossPrice, Digits);
}

// 计算止盈价，可结合风险收益比等因素动态调整（示例简单按固定比例关系，可优化）
double CalculateTakeProfitPrice(double highPrice, double lowPrice, double riskRewardRatio)
{
    double profitRange = (highPrice - lowPrice) * riskRewardRatio;
    return NormalizeDouble(highPrice + profitRange, Digits);
}

// 记录错误日志到文件的简单示例函数（实际应用中可完善日志格式、内容等）
void LogError(const string& errorMessage)
{
    // 这里假设将错误信息写入名为 "ea_error.log" 的文件中，可根据实际情况调整文件路径和写入方式
    FileOpen(1, "ea_error.log", FILE_WRITE | FILE_APPEND);
    FileWrite(1, errorMessage + "\n");
    FileClose(1);
}

// 填充初始历史数据的函数（示例，按照MT5获取历史数据的方式来实现，比如使用CopyRates等函数，并进行数据标准化处理）
void FillInitialHistoricalData(double& pricesArray, double& meanPrice, double& stdDevPrice)
{
    MqlRates ratesBuffer[];
    int copied = CopyRates(Symbol(), Period(), 0, trainingWindowSize, ratesBuffer);
    if (copied < trainingWindowSize)
    {
        LogError("获取初始历史数据失败，数据量不足");
        return;
    }
    // 计算初始均值和标准差用于标准化
    double sum = 0;
    for (int i = 0; i < trainingWindowSize; i++)
    {
        pricesArray[i] = ratesBuffer[i].close;
        sum += ratesBuffer[i].close;
    }
    meanPrice = sum / trainingWindowSize;
    double varianceSum = 0;
    for (int i = 0; i < trainingWindowSize; i++)
    {
        double diff = pricesArray[i] - meanPrice;
        varianceSum += diff * diff;
    }
    stdDevPrice = MathSqrt(varianceSum / trainingWindowSize);
    // 对初始数据进行标准化
    for (int i = 0; i < trainingWindowSize; i++)
    {
        pricesArray[i] = (ratesBuffer[i].close - meanPrice) / stdDevPrice;
    }
}

// 更新历史数据的函数，移除最早的数据，添加最新的收盘价，按照MT5方式获取最新价格，并进行标准化处理
void UpdateHistoricalData(double& pricesArray, double& meanPrice, double& stdDevPrice)
{
    for (int i = 0; i < trainingWindowSize - 1; i++)
    {
        pricesArray[i] = pricesArray[i + 1];
    }
    MqlRates latestRate;
    if (CopyRates(Symbol(), Period(), 0, 1, &latestRate) == 1)
    {
        pricesArray[trainingWindowSize - 1] = (latestRate.close - meanPrice) / stdDevPrice;
    }
    else
    {
        LogError("更新历史数据时获取最新收盘价失败");
    }
}

// 更精细地准备训练数据和标签，结合更多交易相关信息生成标签（示例可改进），同时进行数据标准化处理
void PrepareTrainingDataAndLabels(double historicalPrices[], double trainingLabels[], double& meanPrice, double& stdDevPrice)
{
    // 示例中结合价格变化幅度、趋势等信息生成更有意义的标签，比如价格上涨超过一定幅度设为1，下跌超过一定幅度设为0，波动较小时设为0.5等（可调整）
    // 重新计算均值和标准差，考虑新加入的数据点
    double sum = 0;
    for (int i = 0; i < trainingWindowSize; i++)
    {
        sum += historicalPrices[i];
    }
    meanPrice = sum / trainingWindowSize;
    double varianceSum = 0;
    for (int i = 0; i < trainingWindowSize; i++)
    {
        double diff = historicalPrices[i] - meanPrice;
        varianceSum += diff * diff;
    }
    stdDevPrice = MathSqrt(varianceSum / trainingWindowSize);
    for (int i = 0; i < trainingWindowSize - 1; i++)
    {
        double priceChange = historicalPrices[i + 1] - historicalPrices[i];
        double priceChangePercent = priceChange / historicalPrices[i] * 100;
        if (priceChangePercent > 0.2)
        {
            trainingLabels[i] = 1;
        }
        else if (priceChangePercent < -0.2)
        {
            trainingLabels[i] = 0;
        }
        else
        {
            trainingLabels[i] = 0.5;
        }
        // 对训练数据进行标准化更新
        historicalPrices[i] = (historicalPrices[i] - meanPrice) / stdDevPrice;
    }
}

// 准备当前输入数据，丰富输入特征，不仅仅局限于收盘价（示例可扩展），同时对输入数据进行标准化处理
double** PrepareCurrentInputData(double historicalPrices[], double& meanPrice, double& stdDevPrice)
{
    double** inputData = new double*[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
        inputData[i] = new double[1];
    }
    // 加入移动平均线特征（示例简单计算，可优化），并标准化
    double ma5 = CalculateMovingAverage(historicalPrices, 5, meanPrice, stdDevPrice);
    double ma10 = CalculateMovingAverage(historicalPrices, 10, meanPrice, stdDevPrice);
    // 加入RSI指标特征（示例简单计算，可优化），并标准化
    double rsi = CalculateRSI(historicalPrices, meanPrice, stdDevPrice);
    // 加入ATR指标特征（示例简单计算，可优化），用于体现市场波动情况，在止损止盈等计算中会用到
    double atr = CalculateATR(historicalPrices);
    // 填充输入数据矩阵，可继续添加更多特征
    inputData[0][0] = (historicalPrices[trainingWindowSize - 1] - meanPrice) / stdDevPrice;
    inputData[1][0] = ma5;
    inputData[2][0] = ma10;
    inputData[3][0] = rsi;
    inputData[4][0] = atr;
    // 其他特征填充（示例可继续完善）
    //...

    return inputData;
}

// 简单示例计算移动平均线函数（可优化调整），并考虑数据标准化
double CalculateMovingAverage(double prices[], int period, double& meanPrice, double& stdDevPrice)
{
    double sum = 0;
    int validDataCount = 0;
    for (int i = trainingWindowSize - period; i < trainingWindowSize; i++)
    {
        if (i >= 0)
        {
            sum += prices[i];
            validDataCount++;
        }
    }
    double avg = sum / validDataCount;
    // 标准化处理
    avg = (avg - meanPrice) / stdDevPrice;
    return avg;
}

// 简单示例计算RSI指标函数（可优化调整，此处采用简化常规计算思路），并考虑数据标准化
double CalculateRSI(double prices[], double& meanPrice, double& stdDevPrice)
{
    int n = 14;  // RSI计算周期，可调整
    double gains = 0;
    double losses = 0;
    for (int i = trainingWindowSize - n; i < trainingWindowSize - 1; i++)
    {
        double diff = prices[i + 1] - prices[i];
        if (diff > 0)
        {
            gains += diff;
        }
        else
        {
        losses += (-diff);
    }
    }
    double avgGain = gains / n;
    double avgLoss = losses / n;
    double rs = avgGain / avgLoss;
    double rsi = 100 - 100 / (1 + rs);
    // 标准化处理
    rsi = (rsi - meanPrice) / stdDevPrice;
    return rsi;
}

// 简单示例计算ATR指标函数（可优化调整），用于衡量市场波动情况
double CalculateATR(double prices[])
{
    int period = 14;  // ATR计算周期，可调整
    double trSum = 0;
    for (int i = trainingWindowSize - period; i < trainingWindowSize; i++)
    {
        double highLowDiff = MathAbs(prices[i] - prices[i + 1]);
        double highCloseDiff = MathAbs(prices[i] - prices[i + 2]);
        double lowCloseDiff = MathAbs(prices[i + 1] - prices[i + 2]);
        double tr = MathMax(highLowDiff, MathMax(highCloseDiff, lowCloseDiff));
        trSum += tr;
    }
    return trSum / period;
}

// 开多单函数，优化后结构更清晰，减少重复代码，提高可读性，同时结合更多策略逻辑和新的条件判断，以及考虑动态风险控制
void OpenBuyOrder()
{
    // 先判断是否处于允许开单时段
    if (!IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute))
        return;

    // 结合更多技术指标等综合判断趋势和箱体情况
    double slPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, SLPeriod, 1));
    double slPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, SLPeriod, 1));
    double ma20 = CalculateMovingAverage(historicalPrices, 20, meanPrice, stdDevPrice);  // 新增20周期移动平均线判断
    double rsi = CalculateRSI(historicalPrices, meanPrice, stdDevPrice);
    double atr = CalculateATR(historicalPrices);

    // 趋势箱体相关判断优化，加入更多条件筛选
    bool trendCallbackFlag = false;
    double trendPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, TrendPeriod, 1));
    if (Close[0] - iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, TrendPeriod, 1)) > TrendGap * PointX * Point &&
        Close[0] > ma20 &&  // 价格在20周期移动平均线之上，增强趋势判断
        rsi < 70)  // RSI不过高，避免超买情况开仓
    {
        trendCallbackFlag = true;
    }

    // 震荡箱体相关判断优化，结合新特征和条件
    bool oscCondition = false;
    double oscPeriodHighPrice1 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice1 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double oscPeriodHighPrice2 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double oscPeriodLowPrice2 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 2));
    double kValue = CalculateKValue(oscPeriodHighPrice1, oscPeriodLowPrice1, Close[0]);
    if (oscPeriodHighPrice2 - Close[0] > OSCGap * PointX * Point && kValue < OSCLevel - 100.0 && Close[0] < oscPeriodLowPrice2 && rsi < 60)
    {
        oscCondition = true;
    }

    // 计算止损价和止盈价，考虑更多因素调整，包括市场波动情况（ATR）和风险收益比等
    double riskRewardRatio = 1.5;  // 预设风险收益比，可根据策略调整
    double stopLossPrice = CalculateStopLossPrice(slPeriodLowPrice, Bid, MinSL, MaxSL, PointX, Point, atr);
    double takeProfitPrice = CalculateTakeProfitPrice(slPeriodHighPrice, slPeriodLowPrice, riskRewardRatio);

    // 计算开单手数，优化计算逻辑，考虑更多风险因素，结合账户风险承受能力和市场波动情况
    double openLots = 0.0;
    if (Lots <= 0.0)
    {
        openLots = InitLots;
    }
    else
    {
        double riskFactor = 0.02;  // 新增风险因子，可调整，控制仓位规模基于风险考量
        double maxRiskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * riskFactor;
        double stopLossInPips = (Bid - stopLossPrice) / Point;
        openLots = maxRiskAmount / (stopLossInPips * PointX);
        openLots = MathMin(MarketInfo(Symbol(), 25), MathMax(MarketInfo(Symbol(), 23), openLots));
    }
    openLots = NormalizeDouble(openLots, 2);

    // 统计已开仓的多单数量
    int orderCount = GetOrderCount(MAGIC, 0, Symbol());
    if (orderCount!= 0)
        return;

    // 结合深度学习模型预测结果和置信度等综合判断是否开仓
    if (!(trendCallbackFlag) ||!(oscCondition) || predictionConfidence < 0.6 ||!IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute))
        return;

    // 执行开仓操作，若失败则记录错误日志
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    request.action = TRADE_ACTION_DEAL;
    request.symbol = Symbol();
    request.volume = openLots;
    request.type = ORDER_TYPE_BUY;
    request.price = Ask;
    request.sl = stopLossPrice;
    request.tp = takeProfitPrice;
    request.comment = Author;
    request.magic = MAGIC;
    request.deviation = 3;
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = ORDER_FILLING_RETURN;
    if (!OrderSend(request, result))
    {
        LogError("OpenBuyOrder - OrderSend Error: " + IntegerToString(GetLastError()));
    }
}

// 开空单函数，类似开多单函数进行优化，结合更多策略逻辑和新的条件判断，以及考虑动态风险控制
void OpenSellOrder()
{
    if (!IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute))
        return;

    double slPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, SLPeriod, 1));
    double slPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, SLPeriod, 1));
    double ma20 = CalculateMovingAverage(historicalPrices, 20, meanPrice, stdDevPrice);
    double rsi = CalculateRSI(historicalPrices, meanPrice, stdDevPrice);
    double atr = CalculateATR(historicalPrices);

    bool trendCallbackFlag = false;
    if (iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, TrendPeriod, 1)) - Close[0] > TrendGap * PointX * Point &&
        Close[0] < ma20 &&
        rsi > 30)
    {
        trendCallbackFlag = true;
    }

    bool oscCondition = false;
    double oscPeriodHighPrice1 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice1 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double oscPeriodHighPrice2 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double oscPeriodLowPrice2 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 2));
    double kValue = CalculateKValue(oscPeriodHighPrice1, oscPeriodLowPrice1, Close[0]);
    if (Close[0] - oscPeriodLowPrice2 > OSCGap * PointX * Point && kValue > -(OSCLevel) && Close[0] > oscPeriodHighPrice2 && rsi > 40)
    {
        oscCondition = true;
    }

    double riskRewardRatio = 1.5;
    double stopLossPrice = CalculateStopLossPrice(slPeriodHighPrice, Bid, MinSL, MaxSL, PointX, Point, atr);
    double takeProfitPrice = CalculateTakeProfitPrice(slPeriodLowPrice, slPeriodHighPrice - slPeriodLowPrice, riskRewardRatio);

    double openLots = 0.0;
    if (Lots <= 0.0)
    {
        openLots = InitLots;
    }
    else
    {
        double riskFactor = 0.02;
        double maxRiskAmount = AccountInfoDouble(ACCOUNT_BALANCE) * riskFactor;
        double stopLossInPips = (stopLossPrice - Bid) / Point;
        openLots = maxRiskAmount / (stopLossInPips * PointX);
        openLots = MathMin(MarketInfo(Symbol(), 25), MathMax(MarketInfo(Symbol(), 23), openLots));
    }
    openLots = NormalizeDouble(openLots, 2);

    int orderCount = GetOrderCount(MAGIC, 1, Symbol());
    if (orderCount!= 0)
        return;

    if (!(trendCallbackFlag) ||!(oscCondition) || predictionConfidence < 0.6 ||!IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute))
        return;

    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    request.action = TRADE_ACTION_DEAL;
    request.symbol = Symbol();
    request.volume = openLots;
    request.type = ORDER_TYPE_SELL;
    request.price = Bid;
    request.sl = stopLossPrice;
    request.tp = takeProfitPrice;
    request.comment = Author;
    request.magic = MAGIC;
    request.deviation = 3;
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = ORDER_FILLING_RETURN;
    if (!OrderSend(request, result))
    {
        LogError("OpenSellOrder - OrderSend Error: " + IntegerToString(GetLastError()));
    }
}

// 平仓买（多）单函数，优化循环和逻辑判断，使代码更清晰，结合更多条件判断平仓时机，同时考虑动态风险调整等因素
void CloseBuyOrders()
{
    double oscPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double stopPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double takeProfitPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 2));
    double kValue = CalculateKValue(oscPeriodHighPrice, oscPeriodLowPrice, Close[0]);

    if (kValue == 100.0)
        return;

    // 根据直接平仓水平判断平仓，结合更多指标和条件优化，考虑市场波动情况影响平仓判断
    if (Close[0] > stopPrice && kValue > -(DirectCloseLevel) && CalculateRSI(historicalPrices, meanPrice, stdDevPrice) > 70 && CalculateATR(historicalPrices) > 10)
    {
        int magicNumber = MAGIC;
        int orderType = 0;
        string symbol = Symbol();
        if (CloseOrders(magicNumber, orderType, symbol))
        {
            Print("Buy单根据 直接平仓水平 平仓");
            return;
        }
    }

    // 根据盈利点数判断平仓，同时考虑风险收益比等因素优化，结合动态风险控制调整平仓条件
    if (IsProfitPointToClose(magicNumber, orderType, symbol, ProfitPoint, riskRewardRatio) && CalculateRiskRewardRatio() > 1.5 && CalculateATR(historicalPrices) < 20)
    {
        if (CloseOrders(magicNumber, orderType, symbol))
        {
            Print("Buy单根据 盈利平仓点数 平仓");
        }
    }
}

// 平仓卖（空）单函数，与平仓买（多）单函数类似进行优化，结合更多条件判断平仓时机，同时考虑动态风险调整等因素
void CloseSellOrders()
{
    double oscPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double stopPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double takeProfitPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 2));
    double kValue = CalculateKValue(oscPeriodHighPrice, oscPeriodLowPrice, Close[0]);

    if (kValue == 100.0)
        return;

    // 根据直接平仓水平判断平仓，结合更多指标和条件优化，考虑市场波动情况影响平仓判断
    if (Close[0] < takeProfitPrice && kValue < DirectCloseLevel - 100.0 && CalculateRSI(historicalPrices, meanPrice, stdDevPrice) < 30 && CalculateATR(historicalPrices) > 10)
    {
        int magicNumber = MAGIC;
        int orderType = 1;
        string symbol = Symbol();
        if (CloseOrders(magicNumber, orderType, symbol))
        {
            Print("Sell单根据 直接平仓水平 平仓");
            return;
        }
    }

    // 根据盈利点数判断平仓，同时考虑风险收益比等因素优化，结合动态风险控制调整平仓条件
    if (IsProfitPointToClose(magicNumber, orderType, symbol, ProfitPoint, riskRewardRatio) && CalculateRiskRewardRatio() > 1.5 && CalculateATR(historicalPrices) < 20)
    {
        if (CloseOrders(magicNumber, orderType, symbol))
        {
            Print("Sell单根据 盈利平仓点数 平仓");
        }
    }
}

// 计算类似KDJ指标超买超卖值的K值
double CalculateKValue(double oscPeriodHighPrice, double oscPeriodLowPrice, double closePrice)
{
    double kValue = 100.0;
    if (oscPeriodHighPrice - oscPeriodLowPrice > 0.0)
    {
        kValue = (oscPeriodHighPrice - closePrice) * (-100.0) / (oscPeriodHighPrice - oscPeriodLowPrice);
    }
    return kValue;
}

// 关闭指定条件的订单，返回是否成功关闭所有符合条件的订单，优化错误处理和日志记录
bool CloseOrders(int magicNumber, int orderType, string symbol)
{
    bool allClosed = true;
    for (int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES) && OrderSymbol() == symbol && OrderMagicNumber() == magicNumber && OrderType() == orderType)
        {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            request.action = TRADE_ACTION_SLTP;
            request.symbol = symbol;
            request.volume = OrderLots();
            request.type = OrderType();
            request.price = OrderClosePrice();
            request.magic = magicNumber;
            request.deviation = 0;
            request.type_time = ORDER_TIME_GTC;
            request.type_filling = ORDER_FILLING_RETURN;
            if (!OrderSend(request, result))
            {
                allClosed = false;
                LogError("CloseOrders - OrderSend Error for order " + IntegerToString(i) + ": " + IntegerToString(GetLastError()));
            }
        }
    }
    return allClosed;
}

// 判断是否达到基于盈利点数平仓的条件，优化计算逻辑和考虑更多因素，结合风险收益比等
bool IsProfitPointToClose(int magicNumber, int orderType, string symbol, double profitPoint, double riskRewardRatio)
{
    double totalProfit = 0.0;
    double totalLots = 0.0;
    for (int m = OrdersTotal() - 1; m >= 0; m--)
    {
        if (OrderSelect(m, SELECT_BY_POS, MODE_TRADES) && OrderSymbol() == symbol && OrderMagicNumber() == magicNumber && OrderType() == orderType)
        {
            totalProfit += OrderProfit() + OrderSwap() + OrderCommission();
            totalLots += OrderLots();
        }
    }
    double averageProfit = totalLots > 0.0? totalProfit / totalLots : 0.0;
    return averageProfit > profitPoint * 10.0 && CalculateRiskRewardRatio() > riskRewardRatio;
}

// 新增函数，简单计算当前持仓的风险收益比（示例，可优化完善），考虑更准确的计算方式和市场动态因素
double CalculateRiskRewardRatio()
{
    double totalProfit = 0.0;
    double totalRisk = 0.0;
    for (int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            double entryPrice = OrderOpenPrice();
            double currentPrice = Close[0];
            double stopLossPrice = OrderStopLoss();
            double takeProfitPrice = OrderTakeProfit();
            if (OrderType() == ORDER_TYPE_BUY)
            {
                totalProfit += takeProfitPrice - currentPrice;
                totalRisk += currentPrice - stopLossPrice;
            }
            else
            {
                totalProfit += currentPrice - takeProfitPrice;
                totalRisk += stopLossPrice - currentPrice;
            }
        }
    }
    return totalProfit / totalRisk;
}

// 判断开仓条件的函数，优化逻辑结构和变量命名，使其更易读，结合更多综合因素判断，融入更多市场分析维度
bool IsOpenPositionCondition(int direction)
{
    double oscPeriodHighPrice1 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice1 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double oscPeriodHighPrice2 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double oscPeriodLowPrice2 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double kValue = CalculateKValue(oscPeriodHighPrice1, oscPeriodLowPrice1, Close[0]);

    if (kValue == 100.0)
        return false;

    bool openCondition = false;
    if (direction == 0 && oscPeriodHighPrice2 - Close[0] > OSCGap * PointX * Point && kValue < OSCLevel - 100.0 && Close[0] < oscPeriodLowPrice2 &&
        CalculateRSI(historicalPrices, meanPrice, stdDevPrice) < 60 && CalculateMovingAverage(historicalPrices, 10, meanPrice, stdDevPrice) < Close[0] &&
        CalculateATR(historicalPrices) < 15)  // 增加更多条件判断，如ATR体现的市场波动情况
    {
        openCondition = true;
    }
    else if (direction == 1 && Close[0] - oscPeriodLowPrice2 > OSCGap * PointX * Point && kValue > -(OSCLevel) && Close[0] > oscPeriodHighPrice2 &&
             CalculateRSI(historicalPrices, meanPrice, stdDevPrice) > 40 && CalculateMovingAverage(historicalPrices, 10, meanPrice, stdDevPrice) > Close[0] &&
             CalculateATR(historicalPrices) < 15)
    {
        openCondition = true;
    }

    return openCondition;
}

// 判断是否是有效的开多单信号，综合多种策略逻辑和信号确认机制，进一步优化结合更多因素判断，包括宏观经济数据等外部因素（可拓展部分）
bool IsValidBuySignal()
{
    // 先判断是否处于允许开单时段
    if (!IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute))
        return false;

    // 原有的趋势箱体、震荡箱体等相关条件判断结合新添加的指标和条件综合考量
    double slPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, SLPeriod, 1));
    double slPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, SLPeriod, 1));
    bool callbackFlag = false;
    double trendPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, TrendPeriod, 1));
    if (Close[0] - trendPeriodLowPrice > TrendGap * PointX * Point && Close[0] > CalculateMovingAverage(historicalPrices, 20, meanPrice, stdDevPrice) &&
        CalculateRSI(historicalPrices, meanPrice, stdDevPrice) < 70 && CalculateATR(historicalPrices) < 15)
    {
        callbackFlag = true;
    }

    bool oscCondition = false;
    double oscPeriodHighPrice1 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice1 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double oscPeriodHighPrice2 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double oscPeriodLowPrice2 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 2));
    double kValue = CalculateKValue(oscPeriodHighPrice1, oscPeriodLowPrice1, Close[0]);
    if (oscPeriodHighPrice2 - Close[0] > OSCGap * PointX * Point && kValue < OSCLevel - 100.0 && Close[0] < oscPeriodLowPrice2 &&
        CalculateRSI(historicalPrices, meanPrice, stdDevPrice) < 60 && CalculateATR(historicalPrices) < 10)
    {
        oscCondition = true;
    }

    return callbackFlag && oscCondition && predictionConfidence > 0.6 && IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute);
}

// 判断是否是有效的开空单信号，综合多种策略逻辑和信号确认机制，进一步优化结合更多因素判断，包括宏观经济数据等外部因素（可拓展部分）
bool IsValidSellSignal()
{
    // 先判断是否处于允许开单时段
    if (!IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute))
        return false;
	// 原有的趋势箱体、震荡箱体等相关条件判断结合新添加的指标和条件综合考量
    double slPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, SLPeriod, 1));
    double slPeriodLowPrice = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, SLPeriod, 1));
    bool callbackFlag = false;
    double trendPeriodHighPrice = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, TrendPeriod, 1));
    if (trendPeriodHighPrice - Close[0] > TrendGap * PointX * Point && Close[0] < CalculateMovingAverage(historicalPrices, 20, meanPrice, stdDevPrice) &&
        CalculateRSI(historicalPrices, meanPrice, stdDevPrice) > 30 && CalculateATR(historicalPrices) < 15)
    {
        callbackFlag = true;
    }

    bool oscCondition = false;
    double oscPeriodHighPrice1 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 1));
    double oscPeriodLowPrice1 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 1));
    double oscPeriodHighPrice2 = iHigh(Symbol(), 0, iHighest(Symbol(), 0, MODE_HIGH, OSCPeriod, 2));
    double oscPeriodLowPrice2 = iLow(Symbol(), 0, iLowest(Symbol(), 0, MODE_LOW, OSCPeriod, 2));
    double kValue = CalculateKValue(oscPeriodHighPrice1, oscPeriodLowPrice1, Close[0]);
    if (Close[0] - oscPeriodLowPrice2 > OSCGap * PointX * Point && kValue > -(OSCLevel) && Close[0] > oscPeriodHighPrice2 &&
        CalculateRSI(historicalPrices, meanPrice, stdDevPrice) > 40 && CalculateATR(historicalPrices) < 10)
    {
        oscCondition = true;
    }

    return callbackFlag && oscCondition && predictionConfidence > 0.6 && IsInTradingTime(TimeCurrent() / 3600, StartHour, StartMinute, StopHour, StopMinute);
}
