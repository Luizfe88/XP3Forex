<<<<<<< HEAD
//+------------------------------------------------------------------+
//| calendar_exporter.mq5 - XP3 PRO FOREX v5.2 Land Trading         |
//| Exporta calendário econômico do MT5 para JSON automaticamente  |
//| TIPO: EXPERT ADVISOR (EA) - Arraste para qualquer gráfico      |
//+------------------------------------------------------------------+
#property copyright "XP3 PRO FOREX"
#property version   "5.20"
#property description "Exporta calendário econômico para JSON periodicamente"

input string OutputPath = "..\\..\\..\\xp3forex\\data\\mt5_calendar.json";
input int    UpdateIntervalMinutes = 30; // Intervalo de atualização em minutos
input int    DaysBack   = 1;    // Dias no passado
input int    DaysAhead  = 7;    // Dias à frente

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Cria timer
   EventSetTimer(UpdateIntervalMinutes * 60);
   
   // Executa primeira vez
   ExportCalendar();
   
   Print("✅ Calendar Exporter iniciado via Timer (", UpdateIntervalMinutes, " min)");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer() {
   ExportCalendar();
}

//+------------------------------------------------------------------+
//| Função Principal de Exportação                                   |
//+------------------------------------------------------------------+
void ExportCalendar() {
    MqlCalendarValue values[];
    datetime from = TimeCurrent() - (DaysBack * 86400);
    datetime to = TimeCurrent() + (DaysAhead * 86400);
    
    int count = CalendarValueHistory(values, from, to);
    
    if(count < 0) {
        Print("⚠️ Erro ao obter calendário: ", GetLastError());
        return;
    }
    
    int handle = FileOpen(OutputPath, FILE_WRITE|FILE_TXT|FILE_ANSI);
    if(handle == INVALID_HANDLE) {
        Print("❌ Erro ao abrir arquivo: ", GetLastError());
        return;
    }
    
    FileWriteString(handle, "[\n");
    
    int validEvents = 0;
    for(int i = 0; i < count; i++) {
        MqlCalendarEvent event;
        MqlCalendarCountry country;
        
        if(!CalendarEventById(values[i].event_id, event)) continue;
        if(!CalendarCountryById(event.country_id, country)) continue;
        
        string importance;
        if(event.importance == CALENDAR_IMPORTANCE_HIGH) importance = "High";
        else if(event.importance == CALENDAR_IMPORTANCE_MODERATE) importance = "Medium";
        else importance = "Low";
        
        // Formata data ISO 8601
        string dateStr = TimeToString(values[i].time, TIME_DATE);
        string timeStr = TimeToString(values[i].time, TIME_MINUTES);
        StringReplace(dateStr, ".", "-");
        string isoDate = dateStr + "T" + timeStr + ":00+00:00";
        
        string title = event.name;
        StringReplace(title, "\"", "'");
        StringReplace(title, "\\", ""); // Remove barras invertidas
        
        string separator = (validEvents > 0) ? ",\n" : "";
        string json = StringFormat(
            "%s  {\"date\":\"%s\",\"country\":\"%s\",\"title\":\"%s\",\"impact\":\"%s\"}",
            separator, isoDate, country.currency, title, importance
        );
        
        FileWriteString(handle, json);
        validEvents++;
    }
    
    FileWriteString(handle, "\n]");
    FileClose(handle);
    
    Print("📅 Calendário exportado: ", validEvents, " eventos para ", OutputPath);
}
=======
//+------------------------------------------------------------------+
//| calendar_exporter.mq5 - XP3 PRO FOREX v5.2 Land Trading         |
//| Exporta calendário econômico do MT5 para JSON automaticamente  |
//| TIPO: EXPERT ADVISOR (EA) - Arraste para qualquer gráfico      |
//+------------------------------------------------------------------+
#property copyright "XP3 PRO FOREX"
#property version   "5.20"
#property description "Exporta calendário econômico para JSON periodicamente"

input string OutputPath = "..\\..\\..\\xp3forex\\data\\mt5_calendar.json";
input int    UpdateIntervalMinutes = 30; // Intervalo de atualização em minutos
input int    DaysBack   = 1;    // Dias no passado
input int    DaysAhead  = 7;    // Dias à frente

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Cria timer
   EventSetTimer(UpdateIntervalMinutes * 60);
   
   // Executa primeira vez
   ExportCalendar();
   
   Print("✅ Calendar Exporter iniciado via Timer (", UpdateIntervalMinutes, " min)");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer() {
   ExportCalendar();
}

//+------------------------------------------------------------------+
//| Função Principal de Exportação                                   |
//+------------------------------------------------------------------+
void ExportCalendar() {
    MqlCalendarValue values[];
    datetime from = TimeCurrent() - (DaysBack * 86400);
    datetime to = TimeCurrent() + (DaysAhead * 86400);
    
    int count = CalendarValueHistory(values, from, to);
    
    if(count < 0) {
        Print("⚠️ Erro ao obter calendário: ", GetLastError());
        return;
    }
    
    int handle = FileOpen(OutputPath, FILE_WRITE|FILE_TXT|FILE_ANSI);
    if(handle == INVALID_HANDLE) {
        Print("❌ Erro ao abrir arquivo: ", GetLastError());
        return;
    }
    
    FileWriteString(handle, "[\n");
    
    int validEvents = 0;
    for(int i = 0; i < count; i++) {
        MqlCalendarEvent event;
        MqlCalendarCountry country;
        
        if(!CalendarEventById(values[i].event_id, event)) continue;
        if(!CalendarCountryById(event.country_id, country)) continue;
        
        string importance;
        if(event.importance == CALENDAR_IMPORTANCE_HIGH) importance = "High";
        else if(event.importance == CALENDAR_IMPORTANCE_MODERATE) importance = "Medium";
        else importance = "Low";
        
        // Formata data ISO 8601
        string dateStr = TimeToString(values[i].time, TIME_DATE);
        string timeStr = TimeToString(values[i].time, TIME_MINUTES);
        StringReplace(dateStr, ".", "-");
        string isoDate = dateStr + "T" + timeStr + ":00+00:00";
        
        string title = event.name;
        StringReplace(title, "\"", "'");
        StringReplace(title, "\\", ""); // Remove barras invertidas
        
        string separator = (validEvents > 0) ? ",\n" : "";
        string json = StringFormat(
            "%s  {\"date\":\"%s\",\"country\":\"%s\",\"title\":\"%s\",\"impact\":\"%s\"}",
            separator, isoDate, country.currency, title, importance
        );
        
        FileWriteString(handle, json);
        validEvents++;
    }
    
    FileWriteString(handle, "\n]");
    FileClose(handle);
    
    Print("📅 Calendário exportado: ", validEvents, " eventos para ", OutputPath);
}
>>>>>>> c2c8056f6002bf0f9e0ecc822dfde8a088dc2bcd
