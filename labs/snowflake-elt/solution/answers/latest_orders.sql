select payload:order_id::string    as order_id,
       payload:status::string      as status,
       payload:customer_id::string as customer_id
from raw.order_events
qualify row_number() over (partition by payload:order_id::string order by loaded_at desc) = 1
