#--------------------------------------------------------------------------
# Import libraries --------------------------------------------------------
#--------------------------------------------------------------------------

library(shiny)
library(data.table)
library(randomForest)

#--------------------------------------------------------------------------
# Read in the model -------------------------------------------------------
#--------------------------------------------------------------------------

model_rf <- readRDS("boston_rf_model.rds")

#--------------------------------------------------------------------------
# User Interface  ---------------------------------------------------------
#--------------------------------------------------------------------------


ui <- pageWithSidebar(
  headerPanel("Boston Median House Prices Predictor"),
  
  sidebarPanel(
    
    tags$label(h3("Input Features")),
    
    numericInput(inputId = "1", label = "Crim - Crime rate"),
    numericInput(inputId = "2", label = "Proportion of Residential zone over 25000 sq.ft"),
    numericInput(inputId = "3", label = "Indus - Proportion of Non-Retail business"),
    numericInput(inputId = "4", label = "Nitrogen Oxides Concentration in ppm"),
    numericInput(inputId = "5", label = "Room - Room numbers (average per dwelling)"),
    numericInput(inputId = "6", label = "Age - Proportion of ower"),
    numericInput(inputId = "7", label = "dus - weighted mean of distances to 5 Boston Employment Centres"),
    numericInput(inputId = "8", label = "Rad - index of accessibility to radial highways"),
    numericInput(inputId = "9", label = "tax - full-value property-tax rate per /$10,000"),
    numericInput(inputId = "10", label = "ptratio - pulter-teacher ratio by town"),
    numericInput(inputId = "11", label = "black - proportion of black community by town"),
    numericInput(inputId = "12", label = "lstat - lower status of the population (%)"))
  
  
)



#--------------------------------------------------------------------------
# Server ------------------------------------------------------------------
#--------------------------------------------------------------------------

server <- function(input, output, session) {
  
}


#--------------------------------------------------------------------------
# running the app ---------------------------------------------------------
#--------------------------------------------------------------------------

shinyApp(ui, server)