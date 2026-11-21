(() => {
  const $ = (id) => document.getElementById(id);
  // One controller per kind: a new image request must not abort a video job's status polling (and vice versa).
  const controllers = { image: null, video: null };
  window.cbMediaTab = (kind) => {
    for (const tab of ['chat', 'image', 'video']) {
      const panel = $(`pg-${tab}-panel`);
      if (panel) panel.hidden = kind !== tab;
    }
    document.querySelectorAll('#media-tabs [role="tab"]').forEach((button) => {
      button.setAttribute('aria-selected', String(button.dataset.tab === kind));
      button.dataset.variant = button.dataset.tab === kind ? 'default' : 'outline';
    });
  };
  const init = () => {
    const tabs = $('media-tabs');
    if (!tabs) return;
    const selected = tabs.dataset.selected;
    const kind = ['image', 'video'].find((kind) =>
      Array.from($(`media-${kind}-session`)?.options || []).some((o) => o.value === selected));
    cbMediaTab(kind || (tabs.dataset.chat === 'true' ? 'chat' : tabs.querySelector('[data-tab]').dataset.tab));
  };
  const imageData = (file) => new Promise((resolve, reject) => {
    if (!file) return resolve(undefined);
    if (file.size > 32 * 1024 * 1024) return reject(new Error('输入图像不得超过 32 MiB'));
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('无法读取图像'));
    reader.readAsDataURL(file);
  });
  const json = async (url, options) => {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(typeof data.error === 'string' ? data.error : JSON.stringify(data.error));
    return data;
  };
  window.cbMediaGenerate = async (kind) => {
    controllers[kind]?.abort();
    const current = controllers[kind] = new AbortController();
    const button = $(`media-${kind}-send`), status = $(`media-${kind}-status`), result = $(`media-${kind}-result`);
    button.disabled = true;
    status.textContent = '正在提交…';
    result.replaceChildren();
    try {
      const payload = {
        session_id: Number($(`media-${kind}-session`).value), prompt: $(`media-${kind}-prompt`).value,
        size: $(`media-${kind}-size`).value, image: await imageData($(`media-${kind}-image`).files[0]),
      };
      const seed = $(`media-${kind}-seed`).value;
      if (seed !== '') payload.seed = Number(seed);
      if (kind === 'image') {
        payload.workflow = $('media-workflow').value;
        payload.upscale = $('media-upscale').checked;
        if ($('media-negative').value) payload.negative_prompt = $('media-negative').value;
      } else {
        payload.seconds = Number($('media-seconds').value);
        payload.loras = Array.from(document.querySelectorAll('[data-media-lora]:checked')).map((input) => ({
          name: input.dataset.mediaLora, strength: Number(input.closest('label').querySelector('input[type="number"]').value),
        }));
      }
      const data = await json(`/ui/media/${kind}`, {
        method: 'POST', signal: current.signal,
        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf },
        body: JSON.stringify(payload),
      });
      if (kind === 'image') {
        for (const item of data.data) {
          const image = new Image();
          image.src = `data:image/png;base64,${item.b64_json}`;
          image.alt = payload.prompt;
          image.style.maxWidth = '100%';
          result.append(image);
        }
        status.textContent = '已完成';
        return;
      }
      status.textContent = `已排队 · ${data.id}`;
      for (;;) {
        await new Promise((resolve, reject) => {
          const abort = () => { clearTimeout(timer); reject(new DOMException('已离开页面', 'AbortError')); };
          const timer = setTimeout(() => { current.signal.removeEventListener('abort', abort); resolve(); }, 3000);
          current.signal.addEventListener('abort', abort, { once: true });
        });
        const job = await json(`/ui/media/videos/${data.id}`, { signal: current.signal });
        status.textContent = ({ queued: '排队中', in_progress: '生成中', completed: '已完成', failed: '失败' })[job.status] + ` · ${job.progress}% · ${data.id}`;
        if (job.status === 'failed') throw new Error(job.error);
        if (job.status === 'completed') {
          const video = document.createElement('video');
          video.controls = true;
          video.style.maxWidth = '100%';
          video.src = `/ui/media/videos/${data.id}/content`;
          result.append(video);
          return;
        }
      }
    } catch (error) {
      if (!current.signal.aborted) status.textContent = `请求失败：${error.message}`;
    } finally {
      // A superseded request must not re-enable the button of the request that replaced it.
      if (controllers[kind] === current) {
        controllers[kind] = null;
        button.disabled = false;
      }
    }
  };
  document.addEventListener('DOMContentLoaded', init);
  document.addEventListener('cb:pageEnter', init);
  document.addEventListener('cb:pageLeave', () => Object.values(controllers).forEach((c) => c?.abort()));
})();
