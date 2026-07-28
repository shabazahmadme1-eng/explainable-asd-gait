import React from 'react';

// The Kinetiq mark: a node branching into two paths — motion diverging from a
// single origin, which is conceptually what the model measures.
const BrandMark = ({ className = 'w-9 h-9' }) => (
  <svg viewBox="0 0 32 32" fill="none" className={className} aria-hidden="true">
    <rect width="32" height="32" rx="9" className="fill-ink-900" />
    <path d="M8 22V10" stroke="#7363e8" strokeWidth="2.4" strokeLinecap="round" />
    <path d="M8 16C13 16 13 10 18 10" stroke="#7363e8" strokeWidth="2.4" strokeLinecap="round" />
    <path d="M8 16C13 16 13 22 18 22" stroke="#9184f0" strokeWidth="2.4" strokeLinecap="round" opacity="0.7" />
    <circle cx="22" cy="10" r="2.4" fill="#7363e8" />
    <circle cx="22" cy="22" r="2.4" fill="#9184f0" />
  </svg>
);

export default BrandMark;
