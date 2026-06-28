// Privacy-friendly funnel analytics. No cookies, no PII. Pluggable provider:
//   - Umami      (window.umami.track)      <- recommended, free cloud + custom events
//   - Plausible  (window.plausible)        <- also fine
//   - Beacon     (cfg.beacon URL)          <- POST events to your own collector
//   - else console.debug when cfg.debug
//
// Funnel it captures: app_open -> download_start -> download_complete | download_abandon
//                     -> app_ready -> generate_start -> generate_done | generate_error
// "wasted" visitors = download_start without download_complete (they left mid-download).

let CFG = {};

export function initAnalytics(cfg = {}) { CFG = cfg || {}; }

export function track(event, props = {}) {
  try {
    const p = { ...props };
    if (window.umami && typeof window.umami.track === 'function') window.umami.track(event, p);
    else if (typeof window.plausible === 'function') window.plausible(event, { props: p });
    else if (CFG.beacon && navigator.sendBeacon)
      navigator.sendBeacon(CFG.beacon, JSON.stringify({ event, props: p, ts: Date.now(), ref: location.href }));
    if (CFG.debug) console.debug('[analytics]', event, p);
  } catch { /* analytics must never break the app */ }
}
