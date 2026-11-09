import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Reveal-on-scroll. Adds the `in` class when the element enters the viewport,
 * so the `.reveal` CSS transition fires. `delay` staggers grouped items.
 */
export function useReveal(delay = 0) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    el.style.transitionDelay = `${delay}ms`;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add('in');
          io.unobserve(el);
        }
      },
      { threshold: 0.12, rootMargin: '0px 0px -8% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [delay]);
  return ref;
}

/**
 * Pointer-reactive card: feeds `--mx/--my` (for the `.spotlight` glow) and
 * `--rx/--ry` (for `.tilt`) based on cursor position over the element.
 */
export function useTiltSpotlight(max = 7) {
  const ref = useRef(null);

  const onMouseMove = useCallback((e) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;
    const py = (e.clientY - r.top) / r.height;
    el.style.setProperty('--mx', `${px * 100}%`);
    el.style.setProperty('--my', `${py * 100}%`);
    el.style.setProperty('--ry', `${(px - 0.5) * 2 * max}deg`);
    el.style.setProperty('--rx', `${-(py - 0.5) * 2 * max}deg`);
  }, [max]);

  const onMouseLeave = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.setProperty('--rx', '0deg');
    el.style.setProperty('--ry', '0deg');
  }, []);

  return { ref, onMouseMove, onMouseLeave };
}

/**
 * Count-up animation that starts when the element scrolls into view.
 * Returns [displayValue, ref]. Preserves a prefix/suffix around the number.
 */
export function useCountUp(target, { duration = 1200, decimals = 0 } = {}) {
  const ref = useRef(null);
  const [val, setVal] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const io = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting || started.current) return;
      started.current = true;
      const start = performance.now();
      const tick = (now) => {
        const t = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        setVal(target * eased);
        if (t < 1) requestAnimationFrame(tick);
        else setVal(target);
      };
      requestAnimationFrame(tick);
    }, { threshold: 0.4 });
    io.observe(el);
    return () => io.disconnect();
  }, [target, duration]);

  return [Number(val).toFixed(decimals), ref];
}
