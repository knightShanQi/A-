from __future__ import annotations

import mimetypes

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .database_source import load_env_file

load_env_file()

from .api_service import (
    FRONTEND_DIST_DIR,
    frontend_dist_available,
    load_enhanced_board_payload,
    load_market_backtest_payload,
    load_news_impact_payload,
    load_quick_board_payload,
    load_symbol_detail_payload,
    get_task_status,
    normalize_api_params,
    search_symbols,
    start_market_backtest_task,
    start_rebuild_ranking_task,
)

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")


def create_app() -> FastAPI:
    app = FastAPI(
        title="A-Share Signal Desk API",
        version="0.2.0",
        description="Separated backend API for the rebuilt A-share web terminal.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/board/quick")
    def quick_board(
        ranking_by: str = Query("attention"),
        board_size: int = Query(50, ge=10, le=100),
        horizon_days: int = Query(3),
        positive_return_pct: float = Query(10.0, ge=5.0, le=50.0),
        watchlist_text: str = Query(""),
        security_scope: str = Query("main_board"),
    ) -> dict:
        params = normalize_api_params(
            ranking_by=ranking_by,
            board_size=board_size,
            horizon_days=horizon_days,
            positive_return_pct=positive_return_pct,
            watchlist_text=watchlist_text,
            security_scope=security_scope,
        )
        return load_quick_board_payload(params)

    @app.get("/api/board/enhanced")
    def enhanced_board(
        ranking_by: str = Query("attention"),
        board_size: int = Query(50, ge=10, le=100),
        horizon_days: int = Query(3),
        positive_return_pct: float = Query(10.0, ge=5.0, le=50.0),
        watchlist_text: str = Query(""),
        security_scope: str = Query("main_board"),
    ) -> dict:
        params = normalize_api_params(
            ranking_by=ranking_by,
            board_size=board_size,
            horizon_days=horizon_days,
            positive_return_pct=positive_return_pct,
            watchlist_text=watchlist_text,
            security_scope=security_scope,
        )
        return load_enhanced_board_payload(params)

    @app.get("/api/symbol/{symbol}")
    def symbol_detail(
        symbol: str,
        horizon_days: int = Query(3),
        positive_return_pct: float = Query(10.0, ge=5.0, le=50.0),
        security_scope: str = Query("main_board"),
    ) -> dict:
        params = normalize_api_params(
            horizon_days=horizon_days,
            positive_return_pct=positive_return_pct,
            security_scope=security_scope,
        )
        try:
            return load_symbol_detail_payload(symbol, params)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Symbol detail load failed: {exc}") from exc

    @app.get("/api/symbol/{symbol}/news-impact")
    def symbol_news_impact(
        symbol: str,
        start_date: str | None = Query(None),
        end_date: str | None = Query(None),
        news_limit: int = Query(120, ge=1, le=300),
        horizons: str = Query("1,3,5"),
        include_disclosures: bool = Query(True),
    ) -> dict:
        try:
            parsed_horizons = tuple(
                int(part.strip())
                for part in str(horizons).split(",")
                if part.strip()
            )
            if not parsed_horizons:
                parsed_horizons = (1, 3, 5)
            if any(horizon < 1 or horizon > 20 for horizon in parsed_horizons):
                raise ValueError("horizons must be between 1 and 20 trading days")
            return load_news_impact_payload(
                symbol,
                start_date=start_date,
                end_date=end_date,
                news_limit=news_limit,
                horizons=parsed_horizons,
                include_disclosures=include_disclosures,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"News impact load failed: {exc}") from exc

    @app.get("/api/search")
    def search(query: str = Query("", alias="q"), limit: int = Query(20, ge=1, le=50)) -> dict:
        return {"query": query, "results": search_symbols(query, limit=limit)}

    @app.post("/api/tasks/rebuild-ranking")
    def rebuild_ranking(
        horizon_days: int = Query(3),
        positive_return_pct: float = Query(10.0, ge=5.0, le=50.0),
        security_scope: str = Query("main_board"),
    ) -> dict:
        params = normalize_api_params(
            horizon_days=horizon_days,
            positive_return_pct=positive_return_pct,
            security_scope=security_scope,
        )
        return start_rebuild_ranking_task(params)

    @app.post("/api/tasks/market-backtest")
    def market_backtest_task(
        date_from: str = Query(...),
        date_to: str = Query(...),
        horizon_days: int = Query(3),
        positive_return_pct: float = Query(10.0, ge=5.0, le=50.0),
        strategy_mode: str = Query("all"),
        top_k: int = Query(50, ge=1, le=300),
        force_rebuild: bool = Query(False),
    ) -> dict:
        return start_market_backtest_task(
            date_from=date_from,
            date_to=date_to,
            horizon_days=horizon_days,
            positive_return_pct=positive_return_pct,
            strategy_mode=strategy_mode,
            top_k=top_k,
            force_rebuild=force_rebuild,
        )

    @app.get("/api/backtests/market/latest")
    def latest_market_backtest(result_limit: int = Query(50, ge=0, le=500)) -> dict:
        return load_market_backtest_payload(result_limit=result_limit)

    @app.get("/api/tasks/{task_id}")
    def task_status(task_id: str) -> dict:
        return get_task_status(task_id)

    if frontend_dist_available():
        assets_dir = FRONTEND_DIST_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(FRONTEND_DIST_DIR / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend_spa(full_path: str):
            requested_path = (FRONTEND_DIST_DIR / full_path).resolve()
            if requested_path.is_file() and FRONTEND_DIST_DIR.resolve() in requested_path.parents:
                return FileResponse(requested_path)
            return FileResponse(FRONTEND_DIST_DIR / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def frontend_missing() -> JSONResponse:
            return JSONResponse(
                {
                    "message": "Frontend assets are not built yet. Run npm install && npm run build in frontend, or use npm run dev for the separate frontend server.",
                }
            )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("a_share_predictor.api:app", host="127.0.0.1", port=8000, reload=False)
