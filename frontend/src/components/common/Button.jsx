import React from 'react';
import Spinner from './Spinner';

const Button = ({ children, onClick, disabled, isLoading, variant = 'primary', className = '' }) => {
  const baseStyle = "flex items-center justify-center px-6 py-3 border font-medium rounded-lg transition-all w-full";
  
  const variants = {
    primary: "border-transparent text-white bg-clinical-blue hover:bg-blue-700 shadow-sm",
    secondary: "border-slate-200 text-slate-700 bg-white hover:bg-slate-50",
    danger: "border-transparent text-white bg-red-600 hover:bg-red-700 shadow-sm"
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`${baseStyle} ${variants[variant]} ${disabled || isLoading ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
    >
      {isLoading ? <Spinner text="Processing..." className="text-white" /> : children}
    </button>
  );
};

export default Button;