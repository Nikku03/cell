import { chromium } from 'playwright-core';
const EXE='/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const URL='file:///home/user/cell/outputs/orphan/cell_app.html';
const errs=[];
const b=await chromium.launch({executablePath:EXE,headless:true,args:['--no-sandbox']});
const p=await b.newPage();
p.on('console',m=>{if(m.type()==='error')errs.push('console: '+m.text());});
p.on('pageerror',e=>errs.push('pageerror: '+e.message));
await p.goto(URL,{waitUntil:'networkidle'}); await p.waitForTimeout(500);

console.log('=== load ===');
console.log('title:', await p.title());
console.log('genes listed:', await p.$$eval('#list .gene', a=>a.length));
console.log('metrics:', (await p.textContent('#metrics')).trim());
console.log('WT growth:', await p.textContent('#wt'));

console.log('\n=== knock out pyrB (search + click) ===');
await p.fill('#search','pyrb'); await p.waitForTimeout(200);
await p.click('#list .gene'); await p.waitForTimeout(150);
await p.click('button:has-text("knock out")'); await p.waitForTimeout(200);
console.log('verdict:', (await p.textContent('#verdict')).replace(/\s+/g,' ').trim());
console.log('result:', (await p.textContent('#presult')).replace(/\s+/g,' ').trim().slice(0,90));

console.log('\n=== reset, KO crp, switch to arabinose ===');
await p.click('button:has-text("reset")');
await p.fill('#search','crp'); await p.waitForTimeout(150);
await p.click('#list .gene'); await p.waitForTimeout(150);
await p.click('button:has-text("knock out")'); await p.waitForTimeout(150);
console.log('crp KO on glucose -> verdict:', (await p.textContent('#verdict')).replace(/\s+/g,' ').trim());
await p.selectOption('#cond','arabinose'); await p.waitForTimeout(200);
console.log('crp KO on arabinose -> verdict:', (await p.textContent('#verdict')).replace(/\s+/g,' ').trim());
const res=(await p.textContent('#presult')).replace(/\s+/g,' ').trim();
console.log('  cascade text:', res.slice(0,120));

console.log('\n=== dynamics tab (SOS ODE) ===');
await p.click('.tab[data-t="dyn"]'); await p.waitForTimeout(400);
console.log('dyn svg paths:', await p.$$eval('#dynsvg svg path', a=>a.length));

console.log('\n=== network tab (crp regulon) ===');
await p.fill('#search','crp');
await p.click('.tab[data-t="perturb"]'); await p.waitForTimeout(100);
await p.fill('#search','crp'); await p.waitForTimeout(150);
await p.click('#list .gene'); await p.waitForTimeout(150);
await p.click('button:has-text("view regulon")'); await p.waitForTimeout(300);
console.log('network nodes:', await p.$$eval('#netsvg svg circle', a=>a.length));

await p.screenshot({path:'/home/user/cell/outputs/orphan/cell_app_screenshot.png',fullPage:false});
console.log('\n=== JS ERRORS:', errs.length?errs.join(' | '):'NONE');
await b.close();
