with latest as (
    select payload
    from raw.order_events
    qualify row_number() over (partition by payload:order_id::string order by loaded_at desc) = 1
)
select payload:order_id::string                                      as order_id,
       item.value:sku::string                                        as sku,
       item.value:qty::int                                           as qty,
       (item.value:qty::int * item.value:price::number(10, 2))::number(10, 2) as line_total
from latest, lateral flatten(input => payload:items) item
