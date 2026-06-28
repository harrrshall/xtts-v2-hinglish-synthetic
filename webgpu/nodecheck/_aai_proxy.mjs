// Tiny local proxy so the browser app can transcribe via AssemblyAI without exposing the API key
// in client code (and to sidestep browser CORS). Holds the key server-side, accepts raw WAV bytes
// on POST /transcribe, runs upload -> create (hi) -> poll, returns { text, confidence }.
//   AAI_KEY=<key> node _aai_proxy.mjs            (listens on :8766)
import http from 'node:http';

const KEY = process.env.AAI_KEY;
if (!KEY) { console.error('set AAI_KEY'); process.exit(1); }
const AAI = 'https://api.assemblyai.com/v2';
const H = { authorization: KEY };
const PORT = +(process.env.PORT || 8766);
const CORS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'POST, OPTIONS',
  'access-control-allow-headers': 'content-type',
};

async function transcribe(wavBuf, lang) {
  const up = await fetch(`${AAI}/upload`, { method: 'POST', headers: { ...H, 'content-type': 'application/octet-stream' }, body: wavBuf });
  const { upload_url } = await up.json();
  const cr = await fetch(`${AAI}/transcript`, { method: 'POST', headers: { ...H, 'content-type': 'application/json' }, body: JSON.stringify({ audio_url: upload_url, language_code: lang || 'hi' }) });
  const { id } = await cr.json();
  for (let i = 0; i < 60; i++) {
    const r = await (await fetch(`${AAI}/transcript/${id}`, { headers: H })).json();
    if (r.status === 'completed') return { text: r.text, confidence: r.confidence, audio_duration: r.audio_duration };
    if (r.status === 'error') throw new Error(r.error || 'aai error');
    await new Promise((res) => setTimeout(res, 2000));
  }
  throw new Error('timeout');
}

http.createServer((req, res) => {
  if (req.method === 'OPTIONS') { res.writeHead(204, CORS); return res.end(); }
  if (req.method === 'POST' && req.url.startsWith('/transcribe')) {
    const lang = new URL(req.url, 'http://x').searchParams.get('lang') || 'hi';
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', async () => {
      try {
        const out = await transcribe(Buffer.concat(chunks), lang);
        res.writeHead(200, { ...CORS, 'content-type': 'application/json' });
        res.end(JSON.stringify(out));
      } catch (e) {
        res.writeHead(500, { ...CORS, 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: String(e?.message || e) }));
      }
    });
    return;
  }
  res.writeHead(404, CORS); res.end('not found');
}).listen(PORT, () => console.log(`aai proxy on :${PORT}`));
