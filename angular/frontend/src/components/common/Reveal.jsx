import React from 'react';
import { useReveal } from '../../hooks/useInteractions.js';

const Reveal = ({ as: Tag = 'div', delay = 0, className = '', children, ...rest }) => {
  const ref = useReveal(delay);
  return (
    <Tag ref={ref} className={`reveal ${className}`} {...rest}>
      {children}
    </Tag>
  );
};

export default Reveal;
