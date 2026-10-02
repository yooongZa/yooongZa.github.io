'use strict';
(() => {
 let token = null;
 const status = document.getElementById('status');
 const opener = document.getElementById('open-app');
 const field = document.getElementById('token');
 const label = document.getElementById('token-label');
 const copy = document.getElementById('copy');
 const render = () => {
  const params = new URLSearchParams(location.hash.slice(1));
  const values = params.getAll('token');
  token = !location.search && values.length === 1 && [...params.keys()].length === 1 && /^[0-9a-fA-F]{64}$/.test(values[0]) ? values[0].toLowerCase() : null;
  opener.hidden = field.hidden = label.hidden = copy.hidden = !token;
  opener.removeAttribute('href'); field.value = token || '';
  document.getElementById('copy-status').textContent = '';
  status.textContent = token
   ? '초대가 도착했어요. 앱에서 이름과 동의를 확인하고 가입을 신청해 주세요.'
   : '초대 코드가 없거나 올바르지 않습니다. 운영자에게 받은 전체 초대 링크를 다시 열어 주세요.';
  if (token) opener.href = 'rosary://community-join?token=' + token;
 };
 addEventListener('hashchange', render); render();
 copy.addEventListener('click',async () => {
  if (!token) return;
  const copiedToken = token;
  const note = document.getElementById('copy-status');
  try { await navigator.clipboard.writeText(copiedToken);if (token === copiedToken) note.textContent = '코드를 복사했어요. 앱의 가입 화면에 붙여넣어 주세요.'; }
  catch { if (token === copiedToken) {field.focus();field.select();note.textContent = '아래 코드가 선택됐어요. 길게 눌러 복사한 뒤 앱의 가입 화면에 붙여넣어 주세요.';} }
 });
})();
