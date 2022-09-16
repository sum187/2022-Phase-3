import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

import { ThemeOptions } from "@mui/material/styles/createTheme";
import { CssBaseline } from "@mui/material";
import { StyledEngineProvider } from "@mui/material/styles";
import { createTheme, ThemeProvider } from "@mui/material";

export const themeOptions: ThemeOptions = {
  breakpoints: {
    // Define custom breakpoint values.
    values: {
      xs: 0,
      sm: 320,
      md: 480,
      lg: 768,
      xl: 1024
    }
  }
};

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  
  <React.StrictMode>
    <CssBaseline />
    <StyledEngineProvider injectFirst>
      <ThemeProvider theme={createTheme(themeOptions)}>
        <App />
      </ThemeProvider>
    </StyledEngineProvider>
  </React.StrictMode>

);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
