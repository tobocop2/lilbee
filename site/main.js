// Interactivity for the lilbee landing page: the install-method tabs, the
// per-method "details" disclosures, and the copy-to-clipboard buttons.

(function () {
  'use strict';

  /** Wire the install-method tab strip: click selection and arrow-key navigation. */
  function initInstallTabs() {
    var tablist = document.querySelector('.installtabs');
    if (!tablist) return;
    var tabs = Array.prototype.slice.call(tablist.querySelectorAll('[role="tab"]'));

    function select(tab) {
      tabs.forEach(function (candidate) {
        var active = candidate === tab;
        candidate.setAttribute('aria-selected', active ? 'true' : 'false');
        candidate.tabIndex = active ? 0 : -1;
        var pane = document.getElementById(candidate.getAttribute('aria-controls'));
        if (pane) pane.hidden = !active;
      });
    }

    tablist.addEventListener('click', function (event) {
      var tab = event.target.closest('[role="tab"]');
      if (!tab) return;
      select(tab);
      tab.focus();
    });

    tablist.addEventListener('keydown', function (event) {
      var index = tabs.indexOf(document.activeElement);
      if (index < 0) return;
      var step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      if (step === 0) return;
      event.preventDefault();
      var next = tabs[(index + step + tabs.length) % tabs.length];
      select(next);
      next.focus();
    });
  }

  /** Toggle the per-method "details" notes when their disclosure button is clicked. */
  function initDetailsToggles() {
    document.addEventListener('click', function (event) {
      var toggle = event.target.closest('.notes-toggle');
      if (!toggle) return;
      var open = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', open ? 'false' : 'true');
      var body = document.getElementById(toggle.getAttribute('aria-controls'));
      if (body) body.hidden = open;
    });
  }

  /** Copy a command to the clipboard when its [ COPY ] button is clicked. */
  function initCopyButtons() {
    document.addEventListener('click', function (event) {
      var button = event.target.closest('.copy');
      if (!button) return;
      var row = button.closest('.install');
      var command = row ? row.querySelector('.cmd') : null;
      copyText(command ? command.textContent : '', function () { flashCopied(button); });
    });
  }

  function copyText(text, done) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, done);
      return;
    }
    var area = document.createElement('textarea');
    area.value = text;
    document.body.appendChild(area);
    area.select();
    try { document.execCommand('copy'); } catch (error) { /* clipboard unavailable */ }
    document.body.removeChild(area);
    done();
  }

  function flashCopied(button) {
    if (button.textContent.indexOf('COPIED') !== -1) return;
    var original = button.textContent;
    button.textContent = '[ COPIED ]';
    setTimeout(function () { button.textContent = original; }, 1200);
  }

  initInstallTabs();
  initDetailsToggles();
  initCopyButtons();
})();
