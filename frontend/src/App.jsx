import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import HomePage from './pages/HomePage';
import ReportPage from './pages/ReportPage';
import LearnPage from './pages/LearnPage';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/learn" element={<LearnPage />} />
          <Route path="/report" element={<ReportPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
