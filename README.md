# yongZa의 공부 기록

[블로그](https://yooongza.github.io/blog/)는 GitHub Pages와 Minima 기본 테마를 사용한다.
루트 페이지와 `wrist-rosary/`는 기존 앱 지원 페이지다.

## 작성·발행 기준

[BLOG_GUIDE.md](BLOG_GUIDE.md)에 문체·분량, 원자료 확인, 최신 생성 음성 선택,
Minima 사용, 코드 검증과 공개 배포 절차를 정리했다. 다음 회차를 작성하거나 기존 글을 수정할 때 이 기준을 따른다.

## 글 추가

`_posts/YYYY-MM-DD-제목.md`에 글을 작성한다. 제목의 회차는 01부터 발행 순서대로 붙이며, 현재 01~17편 다음에는 18편을 쓴다. 기존 글의 공개 URL과 파일 경로는 유지한다. 같은 날 여러 글을 올릴 때는 `date`의 시각으로 순서를 정한다.

```yaml
---
layout: post
title: "18. 다음 공부 기록"
date: 2026-09-29 09:00:00 +0900
permalink: /blog/ai-study/18-next-topic/
description: 이번에 공부한 내용
---
```

음성은 주제별 생성 기록을 확인한 최신 파일을 `blog/assets/audio/`에 넣고 글에 연결한다.
코드와 내려받기용 Markdown은 해당 글의 `blog/ai-study/` 폴더에 둔다.
`_posts`가 발행 원본이며, 본문 수정 시 내려받기용 `article.md`도 함께 갱신한다.
이전 정적 HTML 생성 스크립트로 `blog/index.html`이나 글의 `index.html`을 다시 만들지 않는다.

## 로컬 확인

GitHub Pages 의존성과 호환되는 Ruby 3.3 환경을 사용한다.

```sh
bundle install
bundle exec jekyll serve
```

`http://localhost:4000/blog/`에서 확인한다. `main`에 올리면 GitHub Pages가 빌드한다.
Minima의 기본 레이아웃과 스타일을 사용한다. 헤더의 제목 링크는 `/blog/`로 지정하고,
긴 인라인 코드가 모바일 화면 밖으로 넘치지 않도록 줄바꿈만 보완했다.
