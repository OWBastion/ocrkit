from types import SimpleNamespace
from uuid import uuid4

import json

import pytest

from app.jobs import JobConflict, JobFull, OcrJobs
from test_api import _make_context


def store(tmp_path):
    return OcrJobs(tmp_path / 'jobs.sqlite3', _make_context(), 'https://platform.example', 'secret')


def test_durable_accept_dedup_conflict_and_admission_bound(tmp_path, monkeypatch):
    jobs = store(tmp_path)
    job_id = str(uuid4())
    jobs.accept(job_id, b'image')
    jobs = store(tmp_path)
    jobs.accept(job_id, b'image')
    with pytest.raises(JobConflict):
        jobs.accept(job_id, b'changed')
    monkeypatch.setattr('app.jobs.MAX_PENDING', 1)
    with pytest.raises(JobFull):
        jobs.accept(str(uuid4()), b'another')
    with jobs.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1


def test_result_survives_restart_callback_retry_without_recognition(tmp_path, monkeypatch):
    jobs = store(tmp_path)
    job_id = str(uuid4())
    jobs.accept(job_id, b'image')
    calls = []
    def recognize(ctx, image, identifier, **kwargs):
        calls.append(identifier)
        return SimpleNamespace(model_dump=lambda **kw: {'request_id': identifier, 'ok': True})
    monkeypatch.setattr('app.jobs.recognize_payload', recognize)
    assert jobs.recognize_next()
    jobs = store(tmp_path)
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            if len(requests) == 1:
                raise OSError('connection lost')
            class Response:
                status = 204
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return Response()
    monkeypatch.setattr('app.jobs.build_opener', lambda *args: Opener())
    assert jobs.deliver_next()
    with jobs.connect() as db:
        row = db.execute('SELECT image, result, done FROM jobs').fetchone()
        assert row['image'] is None and row['result'] is not None and not row['done']
        db.execute('UPDATE jobs SET next_attempt = 0')
    assert jobs.deliver_next()
    jobs.accept(job_id, b'image')
    assert not jobs.recognize_next()
    assert not jobs.deliver_next()
    assert calls == [job_id]
    assert requests[0].full_url == f'https://platform.example/v1/ocrkit/jobs/{job_id}/result'
    assert requests[0].get_header('Authorization') == 'Bearer secret'
    assert requests[0].get_header('User-agent') == 'OWBastion-OCRKit/1.0'
    assert requests[0].data == requests[1].data
    with jobs.connect() as db:
        row = db.execute('SELECT image, result, done FROM jobs').fetchone()
        assert row['image'] is None and row['result'] is None and row['done']


def test_failure_and_expired_jobs_callback_without_retaining_image(tmp_path, monkeypatch):
    jobs = store(tmp_path)
    monkeypatch.setattr('app.jobs.recognize_payload', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('bad engine')))
    jobs.accept(str(uuid4()), b'image')
    assert jobs.recognize_next()
    jobs.accept(str(uuid4()), b'image')
    with jobs.connect() as db:
        db.execute('UPDATE jobs SET created = created - 1801 WHERE image IS NOT NULL')
    assert jobs.recognize_next()
    with jobs.connect() as db:
        rows = db.execute('SELECT image, result FROM jobs').fetchall()
    assert all(row['image'] is None for row in rows)
    assert {'OCR_RECOGNITION_FAILED', 'OCR_JOB_EXPIRED'} == {json.loads(row['result'])['errorCode'] for row in rows}


@pytest.mark.parametrize('age, expired', [(601, False), (1799, False), (1800, True)])
def test_queue_deadline_allows_thirty_minutes(tmp_path, monkeypatch, age, expired):
    clock = [10_000.0]
    monkeypatch.setattr('app.jobs.time.time', lambda: clock[0])
    jobs = store(tmp_path)
    job_id = str(uuid4())
    jobs.accept(job_id, b'image')
    calls = []
    def recognize(ctx, image, identifier, **kwargs):
        calls.append(identifier)
        return SimpleNamespace(model_dump=lambda **kw: {'request_id': identifier, 'ok': True})
    monkeypatch.setattr('app.jobs.recognize_payload', recognize)
    clock[0] += age

    assert jobs.recognize_next()
    with jobs.connect() as db:
        row = db.execute('SELECT image, result FROM jobs').fetchone()
    assert row['image'] is None
    result = json.loads(row['result'])
    if expired:
        assert result == {'contractVersion': '1', 'errorCode': 'OCR_JOB_EXPIRED'}
        assert not calls
    else:
        assert result['result'] == {'request_id': job_id, 'ok': True}
        assert calls == [job_id]


@pytest.mark.parametrize('origin', ['http://platform.example', 'https://user:secret@platform.example', 'https://platform.example/path', 'https://platform.example?url=other'])
def test_callback_origin_rejects_insecure_or_user_controlled_urls(tmp_path, origin):
    with pytest.raises(ValueError):
        OcrJobs(tmp_path / 'jobs.sqlite3', _make_context(), origin, 'secret')


def test_privacy_retention_bounds_undelivered_evidence(tmp_path):
    jobs = store(tmp_path)
    jobs.accept(str(uuid4()), b'private image')
    with jobs.connect() as db:
        db.execute('UPDATE jobs SET created = created - 86401')
    assert not jobs.deliver_next()
    with jobs.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
    assert b'private image' not in jobs.path.read_bytes()
